# fix_encoding.py
import os
import chardet

def fix_file_encoding(file_path):
   #修复文件编码问题
    try:
        # 检测文件编码
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        if encoding and encoding.lower() != 'utf-8':
            print(f"Converting {file_path} from {encoding} to UTF-8")
            # 用检测到的编码读取文件
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # 用UTF-8编码写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            print(f"{file_path} is already UTF-8 or encoding detection failed")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # 修复项目中的所有Python文件
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                fix_file_encoding(file_path)

if __name__ == '__main__':
    main()
