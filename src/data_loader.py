from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader


def load_all_documents(data_dir: str) -> List[Any] :
    
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data Path : {data_path}")
    document = []
    
    
    ##  Pdf files
    
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files : { [str(f) for f in pdf_files] }")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            document.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file} : {e}")
        
            
            
    ## TXT files
    
    txt_files = list(data_path.glob('**/*.txt'))
    print(f'[DEBUG] Founded {len(txt_files)} TXT files: {[str(f) for f in txt_files]} ')
    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT : {txt_file}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
            document.extend(loaded)
        except Exception as e:
            print(f"[ERROR] failed to load TXT {txt_file} : {e}")
            
    
    
    ## CSV files
    
    
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len((csv_files))} CSV files : {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV : {csv_file}")
        try :
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            document.extend(loaded)
        except Exception as e:
            print(f"[ERROR] failed to load CSV {csv_file} : {e}")
            
            
    
    ## Excel files
    
    
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(xlsx_files)} xlsx files : {[str(f) for f in xlsx_files]}")
    for xlsx_file in xlsx_files:
        print(f"[DEBUG] Loading XLSX : {xlsx_file}")
        try :
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {(len(loaded))} XLSX docs from {xlsx_file}")
            document.extend(loaded)
        except Exception as e :
            print(f"[ERROR] failed to load Excel {xlsx_file} : {e} ")
            
            
    
    ## Word file
    
    
    word_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(word_files)} word files : {[str(f) for f in word_files]}")
    for word_file in word_files:
        print(f"[DEBUG] Loaded WORD : {word_file}")
        try :
            loader = Docx2txtLoader(str(word_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {(len(loaded))} Word docs from {word_file}")
            document.extend(loaded)
        except Exception as e:
            print(f"[ERROR] failed to load Word Document {word_file} : {e}")
            
    
    
    
    ## JSON files
    
    
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] Found {(len(json_files))} json files : {[str(f) for f in json_files]}")
    for json_file in json_files:
        print(f"[DEBUG] Loaded JSON : {(str(json_file))}")
        try:
            loader = JSONLoader(
                str(json_file),
                jq_schema=".",
                text_content=False
                )
            loaded = loader.load()
            print(f"[DEBUG] Loaded {(len(loaded))} JSON from {json_file}")
            document.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON file{json_file} : {e}")
        
        
    print(f"[DEBUG] Total Loaded Documents : {len(document)}")
    
    # Filter out documents with None or non-string page_content
    valid_documents = []
    invalid_count = 0
    for doc in document:
        content = getattr(doc, 'page_content', None)
        if content is not None and isinstance(content, str):
            valid_documents.append(doc)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"[WARNING] Filtered out {invalid_count} documents with invalid page_content (None or non-string)")
    
    print(f"[DEBUG] Valid Documents after filtering : {len(valid_documents)}")
    return valid_documents


if __name__ == '__main__':
    docs = load_all_documents('Research/data/pdf')
    print(f"Loaded {len(docs)} document.")
    print("Example document:", docs[0] if docs else None)
            
    
# def process_all_pdf(pdf_directory):
    
#     all_documents = []
#     pdf_dir = Path(pdf_directory)
    
#     pdf_files = list(pdf_dir.glob('**/*.pdf'))
    
#     print(f"found {len(pdf_files)} PDF files to Process")
    
#     for pdf_files in pdf_files:
#         print(f"\npreprocessing : {pdf_files.name}")
#         try:
#             loader = PyPDFLoader(str(pdf_files))
#             documents = loader.load()
            
#             for doc in documents:
#                 doc.metadata['source_file'] = pdf_files.name
#                 doc.metadata['file_type'] = 'pdf'
                
#             all_documents.extend(documents)
#             print(f" Loaded {len(documents)} pages")
#         except Exception as e:
#             print(f" Error : {e}")
            
#     print(f"\n Total documents loaded : {len(all_documents)}")
#     return all_documents    

# all_pdf_files = process_all_pdf('./data')