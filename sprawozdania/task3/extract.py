import pypdf
import sys
sys.stdout.reconfigure(encoding='utf-8')
reader = pypdf.PdfReader(r'd:\repos\DigitalSignalProcessing\sprawozdania\task3\zad3.pdf')
with open(r'd:\repos\DigitalSignalProcessing\sprawozdania\task3\extracted.txt', 'w', encoding='utf-8') as f:
    for p in reader.pages:
        f.write(p.extract_text() + '\n')
