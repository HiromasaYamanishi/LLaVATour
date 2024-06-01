from preprocess_data import DataPreprocessor
import psutil
import subprocess
import argparse

class TaskManager:
    def __init__(self):
        pass
    
    @staticmethod
    def check_if_script_running(script_name):
        """特定のPythonスクリプトファイル名が実行中かどうかを確認する"""
        for proc in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
            try:
                # プロセス名がpythonかpython.exeであることを確認
                if 'python' in proc.info['name']:
                    # コマンドライン引数に特定のスクリプト名が含まれているか確認
                    if any(script_name in cmd for cmd in proc.info['cmdline']):
                        print(f"{script_name} is running (PID: {proc.info['pid']})")
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        print(f"{script_name} is not running")
        return False
    
    def download_images(self):
        if not TaskManager.check_if_script_running('preprocess_data.py', ['--function', 'download_images', '--ind', '0']):
            subprocess.Popen(["python", "preprocess_data.py", "--function", "download_images", "--ind", "0"])
        if not TaskManager.check_if_script_running('preprocess_data.py', ['--function', 'download_images', '--ind', '1']):
            subprocess.Popen(["python", "preprocess_data.py", "--function", "download_images", "--ind", "1"])
            
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Task manager Utility")
    parser.add_argument('--function', default='download_images', help='Function to run')
    args = parser.parse_args()

    # DataPreprocessorのインスタンスを作成
    task_manager = TaskManager()
    
    # function引数に基づいて対応するメソッドを実行
    if args.function == 'download_images':
        task_manager.download_images()
    # elif args.function == 'retrieve_reviews':
    #     task_manager.retrieve_reviews()
    # elif args.function == 'make_training_data':
    #     task_manager.make_training_data()
    # elif args.function == 'prepare_splitted_reviews':
    #     task_manager.prepare_splitted_reviews()
    # elif args.function == 'make_training_data_with_review':
    #     task_manager.make_training_data_with_review()
    else:
        print(f"Function {args.function} is not recognized.")