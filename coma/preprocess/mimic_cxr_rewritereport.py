import pandas as pd
import requests
import json
import time
import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import math
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../data/rewrite/medical_report_generator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.getLogger().handlers[1].setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class DeepSeekAPIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = 3
        self.timeout = 60
        
    def generate_text(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> Optional[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"call the API (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    logger.info("API call successful")
                    return content
                else:
                    error_msg = f"API call failed: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"wait {wait_time} seconds and retry...")
                        time.sleep(wait_time)
                        
            except requests.exceptions.RequestException as e:
                logger.error(f"request exception (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"wait {wait_time} seconds and retry...")
                    time.sleep(wait_time)
            except Exception as e:
                logger.error(f"unknown error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
        logger.error("All retry attempts have failed")
        return None


class MedicalReportGenerator:
    def __init__(self, api_key: str, max_workers: int = 8):
        self.api_client = DeepSeekAPIClient(api_key)
        self.max_workers = max_workers
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.empty_files_count = 0
        self.retry_count = 0
        self.lock = threading.Lock()
        
    def load_data_from_csv(self, csv_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['template', 'target_path']
            
            # 检查必要列是否存在
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"missing necessary columns: {missing_columns}")
            
            logger.info(f"successfully loaded data, total {len (df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"failed to load CSV file: {e}")
            raise
    
    def check_empty_files(self, result_df: pd.DataFrame) -> pd.DataFrame:
        empty_files = []
        
        for _, row in result_df.iterrows():
            if row['success']:
                target_path = row['target_path']
                if os.path.exists(target_path):
                    file_size = os.path.getsize(target_path)
                    if file_size == 0:
                        empty_files.append({
                            'row_index': row['row_index'],
                            'target_path': target_path,
                            'template': row['template'],
                            'file_size': file_size,
                            'error_message': 'empty file'
                        })
                    else:
                        try:
                            with open(target_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if not content:
                                    empty_files.append({
                                        'row_index': row['row_index'],
                                        'target_path': target_path,
                                        'template': row['template'],
                                        'file_size': file_size,
                                        'error_message': 'file content is empty'
                                    })
                        except Exception as e:
                            empty_files.append({
                                'row_index': row['row_index'],
                                'target_path': target_path,
                                'template': row['template'],
                                'file_size': file_size,
                                'error_message': f'fail to load file: {e}'
                            })
        
        empty_df = pd.DataFrame(empty_files)
        self.empty_files_count = len(empty_df)
        
        if len(empty_df) > 0:
            logger.warning(f"find {len(empty_df)} empty files:")
            for _, row in empty_df.iterrows():
                logger.warning(f"  row {row['row_index']}: {row['target_path']} - {row['error_message']}")
        
        return empty_df
    
    def get_items_to_process(self, df: pd.DataFrame) -> pd.DataFrame:
        items_to_process = []
        
        for index, row in df.iterrows():
            target_path = row['target_path']
            if not os.path.exists(target_path):
                items_to_process.append({
                    'row_index': index,
                    'template': row['template'],
                    'target_path': target_path,
                    'reason': 'file does not exist'
                })
            else:
                file_size = os.path.getsize(target_path)
                if file_size == 0:
                    items_to_process.append({
                        'row_index': index,
                        'template': row['template'],
                        'target_path': target_path,
                        'reason': 'empty file'
                    })
                else:
                    try:
                        with open(target_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if not content:
                                items_to_process.append({
                                    'row_index': index,
                                    'template': row['template'],
                                    'target_path': target_path,
                                    'reason': 'file content is empty'
                                })
                    except Exception as e:
                        items_to_process.append({
                            'row_index': index,
                            'template': row['template'],
                            'target_path': target_path,
                            'reason': f'fail to load file: {e}'
                        })
        
        items_df = pd.DataFrame(items_to_process)
        
        reason_counts = items_df['reason'].value_counts()
        reason_str = ", ".join([f"{reason}: {count}" for reason, count in reason_counts.items()])
        
        logger.info(f"items need to be processed: {len(items_df)} ({reason_str})")
        
        return items_df
    
    def check_all_files_exist_and_valid(self, df: pd.DataFrame) -> bool:
        all_files_exist_and_valid = True
        
        for index, row in df.iterrows():
            target_path = row['target_path']
            
            if not os.path.exists(target_path):
                all_files_exist_and_valid = False
                break
            file_size = os.path.getsize(target_path)
            if file_size == 0:
                all_files_exist_and_valid = False
                break
            
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        all_files_exist_and_valid = False
                        break
            except Exception:
                all_files_exist_and_valid = False
                break
        
        return all_files_exist_and_valid
    
    def ensure_directory_exists(self, file_path: str) -> None:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"create directory: {directory}")
    
    def save_report(self, content: str, file_path: str) -> bool:
        try:
            self.ensure_directory_exists(file_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"success to save report: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"failed to save report {file_path}: {e}")
            return False
    
    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        row_index = item['row_index']
        template = item['template']
        target_path = item['target_path']
        reason = item['reason']
        
        logger.info(f"processing data in row {row_index} (reason: {reason})")
        
        if reason != 'file does not exist' and os.path.exists(target_path):
            try:
                os.remove(target_path)
                logger.info(f"problematic files have been deleted: {target_path}")
            except Exception as e:
                logger.error(f"failed to delete file {target_path}: {e}")
        
        generated_report = self.api_client.generate_text(template)
        
        result = {
            'row_index': row_index,
            'template': template,
            'target_path': target_path,
            'success': False,
            'generated_report': generated_report,
            'error_message': None,
            'reason': reason
        }
        
        if generated_report is None:
            result['error_message'] = "failed to call API"
            with self.lock:
                self.failed_count += 1
            logger.error(f"processing failed on line {row_index}: failed to call API")
            return result
        
        if self.save_report(generated_report, target_path):
            result['success'] = True
            with self.lock:
                self.processed_count += 1
            logger.info(f"processing successed on line {row_index}")
        else:
            result['error_message'] = "failed to save file"
            with self.lock:
                self.failed_count += 1
            logger.error(f"processing failed on line {row_index}: failed to save file")
        
        return result
    
    def process_items_batch(self, items_to_process: pd.DataFrame, batch_delay: float = 0.1, batch_num: int = 1) -> pd.DataFrame:
        if len(items_to_process) == 0:
            return pd.DataFrame()
        
        results = []
        items_list = []
        for _, row in items_to_process.iterrows():
            items_list.append({
                'row_index': row['row_index'],
                'template': row['template'],
                'target_path': row['target_path'],
                'reason': row['reason']
            })
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {
                executor.submit(self.process_single_item, item): item 
                for item in items_list
            }
            
            with tqdm(total=len(future_to_item), desc=f"process batch  {batch_num}") as pbar:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"processing failed on line {item['row_index']}: {e}")
                        results.append({
                            'row_index': item['row_index'],
                            'template': item['template'],
                            'target_path': item['target_path'],
                            'success': False,
                            'generated_report': None,
                            'error_message': str(e),
                            'reason': item['reason']
                        })
                        with self.lock:
                            self.failed_count += 1
                    
                    pbar.update(1)
                    time.sleep(batch_delay)
        
        result_df = pd.DataFrame(results)
        
        success_count = len(result_df[result_df['success'] == True])
        fail_count = len(result_df[result_df['success'] == False])
        success_rate = (success_count / len(result_df)) * 100 if len(result_df) > 0 else 0
        
        logger.info(f"processing successed on batch {batch_num}: {success_count}/{len(result_df)} ({success_rate:.1f}%)")
        
        return result_df
    
    def process_dataframe(self, df: pd.DataFrame, batch_delay: float = 0.1, max_rounds: int = 10) -> pd.DataFrame:
        if self.check_all_files_exist_and_valid(df):
            logger.info("all files already exist and are not empty, no processing required")
            
            result_df = pd.DataFrame()
            for index, row in df.iterrows():
                result_df = pd.concat([result_df, pd.DataFrame([{
                    'row_index': index,
                    'template': row['template'],
                    'target_path': row['target_path'],
                    'success': True,
                    'generated_report': None,
                    'error_message': None,
                    'reason': 'file already exists and is not empty'
                }])], ignore_index=True)
            
            return result_df
        
        all_results = []
        round_num = 0
        result_df = pd.DataFrame(columns=['row_index', 'template', 'target_path', 'success', 
                                         'generated_report', 'error_message', 'reason'])
        
        while round_num < max_rounds:
            round_num += 1
            logger.info(f"processing start on round {round_num}")
            
            items_to_process = self.get_items_to_process(df)
            
            if len(items_to_process) == 0:
                logger.info(f"round {round_num}: no files need to be processed, complete")
                break
                
            logger.info(f"round {round_num}: {len(items_to_process)} entry need to be processed")
            
            batch_results = self.process_items_batch(items_to_process, batch_delay, round_num)
            
            if len(batch_results) > 0:
                for _, result_row in batch_results.iterrows():
                    row_index = result_row['row_index']
                    
                    existing_index = result_df[result_df['row_index'] == row_index].index
                    
                    if len(existing_index) > 0:
                        result_df.loc[existing_index[0]] = result_row
                    else:
                        result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)
            
            empty_files_df = self.check_empty_files(batch_results)
            empty_count = len(empty_files_df)
            
            if empty_count > 0:
                self.empty_files_count += empty_count
                logger.warning(f"after round {round_num} found {empty_count} empty files")
            
            success_count = len(result_df[result_df['success'] == True])
            total_count = len(df)
            success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
            
            logger.info(f"round {round_num}: success {success_count}/{total_count} ({success_rate:.1f}%)")
            
            if success_count == total_count:
                logger.info("All entries have been successfully processed, ended prematurely")
                break
        
        final_success_count = len(result_df[result_df['success'] == True])
        final_fail_count = len(result_df[result_df['success'] == False])
        final_total_count = len(df)
        final_success_rate = (final_success_count / final_total_count) * 100 if final_total_count > 0 else 0
        
        logger.info(f"final processing result: Success {final_success_count}/{final_total_count} ({final_success_rate:.1f}%)")
        logger.info(f"final processing rounds: {round_num}")
        
        missing_rows = set(df.index) - set(result_df['row_index'])
        if missing_rows:
            logger.warning(f"missing {len(missing_rows)} rows in result, added as a successful status")
            for row_index in missing_rows:
                row = df.loc[row_index]
                result_df = pd.concat([result_df, pd.DataFrame([{
                    'row_index': row_index,
                    'template': row['template'],
                    'target_path': row['target_path'],
                    'success': True,
                    'generated_report': None,
                    'error_message': None,
                    'reason': 'file already exists and not empty'
                }])], ignore_index=True)
        
        return result_df
    
    def generate_summary_report(self, result_df: pd.DataFrame, output_path: str = "processing_summary.json"):
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_processed': len(result_df),
            'successful': len(result_df[result_df['success'] == True]),
            'failed': len(result_df[result_df['success'] == False]),
            'empty_files_found': self.empty_files_count,
            'success_rate': (len(result_df[result_df['success'] == True]) / len(result_df)) * 100 if len(result_df) > 0 else 0,
            'failed_details': result_df[result_df['success'] == False][['row_index', 'error_message', 'reason']].to_dict('records')
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"processing summary save to: {output_path}")
        except Exception as e:
            logger.error(f"failed to save processing summary: {e}")


def main():
    API_KEY = "your_deepseek_api_key_here"
    CSV_FILE_PATH = "../../data/rewrite/medical_reports_data.csv"
    MAX_WORKERS = 8
    BATCH_DELAY = 0.1
    MAX_ROUNDS = 10
    
    try:
        generator = MedicalReportGenerator(api_key=API_KEY, max_workers=MAX_WORKERS)
        
        logger.info(f"load data from csv file: {CSV_FILE_PATH}")
        df = generator.load_data_from_csv(CSV_FILE_PATH)
        result_df = generator.process_dataframe(df, batch_delay=BATCH_DELAY, max_rounds=MAX_ROUNDS)
        generator.generate_summary_report(result_df)
        
        result_csv_path = "../../data/rewrite/detailed_processing_results.csv"
        result_df.to_csv(result_csv_path, index=False, encoding='utf-8')
        logger.info(f"detailed results have been saved to: {result_csv_path}")
        
        failed_cases = result_df[result_df['success'] == False]
        if len(failed_cases) > 0:
            logger.warning(f"final failed cases {len(failed_cases)}:")
            for _, case in failed_cases.iterrows():
                logger.warning(f"  row {case['row_index']}: {case['error_message']} (reason: {case['reason']})")
        else:
            logger.info("all files processed successfully")
        
    except Exception as e:
        logger.error(f"program execution failed: {e}")


def create_sample_csv():
    """创建示例CSV文件（用于测试）"""
    sample_data = {
        'template': [
            """<requirement>
基于以下示例，将医学影像描述报告分解为由若干子句组成的段落。要求将每个独立的医学概念分解为不同的子句。
</requirement>
<example>
原始报告:
心脏大小正常。双肺清晰，未见实质性病变。
分解后的报告:
心脏大小正常。双肺清晰。双肺未见实质性病变。
</example>
<target>
请分解以下报告:
胸部CT示双肺多发微小结节，最大直径约0.3cm。纵隔内未见肿大淋巴结。心脏大小形态正常。
</target>""",
            """<requirement>
基于以下示例，将医学影像描述报告分解为由若干子句组成的段落。要求将每个独立的医学概念分解为不同的子句。
</requirement>
<example>
原始报告:
颅脑CT平扫显示脑实质未见异常密度影。脑室系统大小形态正常。
分解后的报告:
颅脑CT平扫显示脑实质未见异常密度影。脑室系统大小形态正常。
</example>
<target>
请分解以下报告:
肝脏形态大小正常，表面光滑，肝内胆管无扩张。脾脏不大。双肾形态正常，未见结石及积水。
</target>"""
        ],
        'target_path': [
            "output/reports/report_1_decomposed.txt",
            "output/reports/report_2_decomposed.txt"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv("sample_medical_reports.csv", index=False, encoding='utf-8')
    print("示例CSV文件已创建: sample_medical_reports.csv")


if __name__ == "__main__":
    # 如果需要创建示例文件，取消下面的注释
    # create_sample_csv()
    
    main()