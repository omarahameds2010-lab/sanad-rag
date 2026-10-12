"""Batch Processor for Mega Islamic Books & Manuscripts
Converts EPUB collections to SANAD RAG knowledge library format
Automatically shards output to stay under ingestion size limits (200KB per file)
"""

import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Configuration
MAX_CHARS_PER_SHARD = 180000  # Safe under 200KB ingestion limit
INPUT_FOLDER = r'C:\Users\pc\Desktop\sanad-rag\sanad-rag\sanad-rag\raw\epub_books'
OUTPUT_FOLDER = r'C:\Users\pc\Desktop\sanad-rag\sanad-rag\sanad-rag\data\knowledge\mega_libraries'


def extract_text_from_epub(epub_path):
    """Extracts text from EPUB file, returns list of lines."""
    book = ebooklib.epub.read_epub(epub_path)
    lines = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            try:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text().strip()
                if text:
                    for line in text.split('\n'):
                        line_stripped = line.strip()
                        if line_stripped:
                            lines.append(line_stripped)
            except Exception:
                continue
    return lines


def process_mega_books(input_folder, output_folder, max_chars=MAX_CHARS_PER_SHARD):
    """Main processor: iterates over all EPUB files in folder,
    extracts text, and generates sharded output files."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    total_books = 0
    total_shards = 0
    total_lines_processed = 0
    
    epub_files = [f for f in os.listdir(input_folder) if f.endswith('.epub')]
    epub_files.sort()
    
    print(f"Found {len(epub_files)} EPUB files to process\n")
    
    for epub_file in epub_files:
        epub_path = os.path.join(input_folder, epub_file)
        book_title = os.path.splitext(epub_file)[0]
        
        print(f"Processing: {epub_file}")
        print(f"  Book title: {book_title}")
        
        # Extract text data
        lines = extract_text_from_epub(epub_path)
        print(f"  Extracted {len(lines)} text lines")
        
        if not lines:
            print("  No extractable text found, skipping...")
            continue
        
        # Shard and write output
        current_shard = []
        current_chars = 0
        shard_count = 1
        book_lines = 0
        
        base_output_name = f"gen_{book_title}"
        
        for line in lines:
            # Encode line to get accurate char count
            line_encoded = line.encode('utf-8')
            line_len = len(line_encoded)
            
            # Check if adding this line would exceed the limit
            if current_chars + line_len > max_chars and current_shard:
                # Save current shard
                shard_filename = f"{base_output_name}_p{shard_count}.txt"
                shard_path = os.path.join(output_folder, shard_filename)
                
                with open(shard_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(current_shard) + '\n')
                
                total_shards += 1
                book_lines += len(current_shard)
                print(f"  Shard {shard_count}: {len(current_shard)} lines → {shard_filename}")
                
                shard_count += 1
                current_shard = []
                current_chars = 0
            
            current_shard.append(line + '\n')
            current_chars += line_len
            book_lines += 1
        
        # Save the final shard for this book
        if current_shard:
            shard_filename = f"{base_output_name}_p{shard_count}.txt"
            shard_path = os.path.join(output_folder, shard_filename)
            
            with open(shard_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(current_shard) + '\n')
            
            total_shards += 1
            book_lines += len(current_shard)
            print(f"  Final shard {shard_count}: {len(current_shard)} lines → {shard_filename}")
        
        total_books += 1
        total_lines_processed += book_lines
        print(f"Completed: {book_title} ({book_lines} lines)\n")
    
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Books processed: {total_books}")
    print(f"Total shards generated: {total_shards}")
    print(f"Total lines processed: {total_lines_processed}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Max chars per shard: {MAX_CHARS_PER_SHARD} (safe under 200KB ingestion limit)")
    print("=" * 60)


if __name__ == "__main__":
    print("SANAD RAG: Mega Books Batch Processor")
    print("=" * 60)
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Size limit per shard: {MAX_CHARS_PER_SHARD} chars")
    print()
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder not found: {INPUT_FOLDER}")
        print("Please create the folder and add your EPUB files.")
    else:
        process_mega_books(INPUT_FOLDER, OUTPUT_FOLDER, MAX_CHARS_PER_SHARD)