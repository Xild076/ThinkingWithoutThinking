"""PDF utilities for reading and creating PDF documents."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    try:
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = []
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                text_content.append(f"--- Page {page_num + 1} ---\n{text}")
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Successfully extracted text from PDF: {pdf_path} ({len(pdf_reader.pages)} pages)")
            return full_text
    
    except ImportError:
        error_msg = "PyPDF2 not installed. Install with: pip install PyPDF2"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"
    
    except Exception as e:
        error_msg = f"Failed to extract text from PDF: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


def create_pdf_from_text(text: str, output_path: str, title: Optional[str] = None) -> bool:
    """
    Create a PDF file from text content.
    
    Args:
        text: Text content to write to PDF
        output_path: Path where PDF should be saved
        title: Optional title for the PDF
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
        
        # Create the PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
        
        # Add title if provided
        if title:
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor='#1a1a1a',
                spaceAfter=30,
                alignment=TA_CENTER
            )
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 12))
        
        # Split text into paragraphs and add to story
        paragraphs = text.split('\n\n')
        for para_text in paragraphs:
            if para_text.strip():
                # Check if it's a heading (starts with # or is all caps short line)
                if para_text.startswith('#'):
                    # Markdown heading
                    heading_text = para_text.lstrip('#').strip()
                    level = min(len(para_text) - len(para_text.lstrip('#')), 3)
                    style_name = f'Heading{level}'
                    story.append(Paragraph(heading_text, styles[style_name]))
                elif len(para_text) < 60 and para_text.isupper():
                    # All caps heading
                    story.append(Paragraph(para_text, styles['Heading2']))
                else:
                    # Regular paragraph
                    story.append(Paragraph(para_text, styles['Justify']))
                
                story.append(Spacer(1, 12))
        
        # Build the PDF
        doc.build(story)
        logger.info(f"Successfully created PDF: {output_path}")
        return True
    
    except ImportError:
        error_msg = "reportlab not installed. Install with: pip install reportlab"
        logger.error(error_msg)
        return False
    
    except Exception as e:
        error_msg = f"Failed to create PDF: {str(e)}"
        logger.error(error_msg)
        return False


def create_pdf_from_markdown(markdown_text: str, output_path: str, title: Optional[str] = None) -> bool:
    """
    Create a PDF file from markdown-formatted text.
    
    Args:
        markdown_text: Markdown-formatted text content
        output_path: Path where PDF should be saved
        title: Optional title for the PDF
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
        import re
        
        # Create the PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        story = []
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
        styles.add(ParagraphStyle(name='Code', 
                                 fontName='Courier',
                                 fontSize=10,
                                 leftIndent=20,
                                 rightIndent=20,
                                 spaceAfter=12))
        
        # Add title if provided
        if title:
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor='#1a1a1a',
                spaceAfter=30,
                alignment=TA_CENTER
            )
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 12))
        
        # Process markdown
        lines = markdown_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Headings
            if line.startswith('#'):
                level = min(len(line) - len(line.lstrip('#')), 6)
                heading_text = line.lstrip('#').strip()
                style_name = f'Heading{min(level, 3)}'
                story.append(Paragraph(heading_text, styles[style_name]))
                story.append(Spacer(1, 12))
            
            # Bullet lists
            elif line.startswith('- ') or line.startswith('* '):
                list_items = []
                while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                    item_text = lines[i].strip()[2:]
                    list_items.append(ListItem(Paragraph(item_text, styles['Normal'])))
                    i += 1
                story.append(ListFlowable(list_items, bulletType='bullet'))
                story.append(Spacer(1, 12))
                continue
            
            # Code blocks
            elif line.startswith('```'):
                i += 1
                code_lines = []
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                code_text = '\n'.join(code_lines)
                story.append(Paragraph(code_text.replace('\n', '<br/>'), styles['Code']))
                story.append(Spacer(1, 12))
            
            # Regular paragraph
            else:
                # Handle inline markdown (bold, italic, code)
                processed_line = line
                processed_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', processed_line)
                processed_line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', processed_line)
                processed_line = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', processed_line)
                
                story.append(Paragraph(processed_line, styles['Justify']))
                story.append(Spacer(1, 12))
            
            i += 1
        
        # Build the PDF
        doc.build(story)
        logger.info(f"Successfully created PDF from markdown: {output_path}")
        return True
    
    except ImportError as e:
        error_msg = f"Missing dependency: {str(e)}. Install with: pip install reportlab"
        logger.error(error_msg)
        return False
    
    except Exception as e:
        error_msg = f"Failed to create PDF from markdown: {str(e)}"
        logger.error(error_msg)
        return False
