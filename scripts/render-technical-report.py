"""Render the public English technical report using the document runtime reportlab dependency."""
from pathlib import Path
import re, html
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("LubanCJK", "/System/Library/Fonts/STHeiti Light.ttc", subfontIndex=0))
root=Path(__file__).resolve().parents[1]/'papers/technical-report-v0.1'
source=root/'paper/Luban_Technical_Report.md'; output=source.with_suffix('.pdf')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle',fontName='Helvetica-Bold',fontSize=23,leading=28,textColor=HexColor('#12233d'),spaceAfter=18))
styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9,textColor=HexColor('#243247')))
styles.add(ParagraphStyle(name='ReportH2',fontName='Helvetica-Bold',fontSize=16,leading=20,spaceBefore=17,spaceAfter=10,keepWithNext=True,textColor=HexColor('#12233d')))
styles.add(ParagraphStyle(name='ReportH3',fontName='Helvetica-Bold',fontSize=11,leading=15,spaceBefore=10,spaceAfter=7,keepWithNext=True,textColor=HexColor('#ab1830')))
styles.add(ParagraphStyle(name='ReportCell',fontName='Helvetica',fontSize=8,leading=11,textColor=HexColor('#243247')))

def inline(s):
 s=s.replace('中文补充','Chinese supplement')
 s=html.escape(s)
 def link(m):
  dest=m.group(2)
  if not dest.startswith('https:'):
   dest='https://github.com/MaiHHConnect/lubanllm/blob/main/'+('docs/'+dest[len('../docs/'):] if dest.startswith('../docs/') else 'paper/'+dest)
  return f'<link href="{dest}" color="#a81932">{m.group(1)}</link>'
 s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,s)
 s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
 s=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',s)
 s=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',s)
 s=re.sub(r"([\u3400-\u9fff]+)",r'<font name="LubanCJK">\1</font>',s)
 return s

lines=source.read_text().splitlines();story=[];i=0
while i<len(lines):
 line=lines[i].strip();i+=1
 if not line:continue
 if line.startswith('|'):
  rows=[line]
  while i<len(lines) and lines[i].strip().startswith('|'):
   rows.append(lines[i].strip());i+=1
  cells=[[Paragraph(inline(c.strip()),styles['ReportCell']) for c in r.strip('|').split('|')] for r in rows if not re.fullmatch(r'[| :\-]+',r)]
  table=Table(cells,colWidths=[120,64,80,211],repeatRows=1,hAlign='LEFT')
  table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e8edf3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,-1),.4,HexColor('#d3dce5'))]))
  story.extend([table,Spacer(1,12)]);continue
 if line.startswith('# '):style='ReportTitle';line=line[2:]
 elif line.startswith('## '):style='ReportH2';line=line[3:]
 elif line.startswith('### '):style='ReportH3';line=line[4:]
 else:style='ReportBody'
 story.append(Paragraph(inline(line),styles[style]))

def page_frame(canvas,doc):
 canvas.saveState();w,h=doc.pagesize
 canvas.setStrokeColor(HexColor('#d3dce5'));canvas.line(60,h-43,w-60,h-43)
 canvas.setFont('Helvetica',8);canvas.setFillColor(HexColor('#61718a'))
 canvas.drawString(60,h-33,'LUBAN ZERO  /  TECHNICAL REPORT v0.2')
 canvas.drawRightString(w-60,h-33,'RESEARCH PREVIEW')
 canvas.drawString(60,32,'Report v0.2: 23 October 2026  |  Not peer reviewed')
 canvas.drawRightString(w-60,32,str(doc.page));canvas.restoreState()
SimpleDocTemplate(str(output),pagesize=(595.28,841.89),rightMargin=60,leftMargin=60,topMargin=65,bottomMargin=55,title='Luban: Budget-Constrained Continual Learning with Disk-Resident Neural Experts',author='vinson (麦好火电商研究)',subject='Technical Report v0.2 - Research Preview').build(story,onFirstPage=page_frame,onLaterPages=page_frame)
print(output)
