from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import PdfFormatter

# Read the Python script file
with open('HW2.py', 'r') as f:
    code = f.read()

# Create a PDF document
doc = SimpleDocTemplate("output.pdf", pagesize=letter)

# Create a paragraph style
styles = getSampleStyleSheet()
paragraph_style = styles['Normal']

# Highlight the Python code
highlighted_code = highlight(code, PythonLexer(), PdfFormatter())

# Add the highlighted code to the PDF
formatted_code = Paragraph(highlighted_code, paragraph_style)
doc.build([formatted_code])
