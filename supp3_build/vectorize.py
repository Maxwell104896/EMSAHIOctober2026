import json,re,xml.etree.ElementTree as ET
import numpy as np
def export_native_geometry(fig,stem,jsonpath):
 fig.savefig(str(stem)+".png",dpi=600);fig.savefig(str(stem)+".pdf");fig.savefig(str(stem)+".svg")
 # Save geometry and text as editable PowerPoint primitives.
 fig.canvas.draw();renderer=fig.canvas.get_renderer();texts=[];W,H=fig.get_size_inches()*72
 ns='{http://www.w3.org/2000/svg}';tree=ET.parse(str(stem)+'.svg');root=tree.getroot();defs={n.attrib['id']:n for n in root.iter() if 'id' in n.attrib};paths=[]
 def style(s):return dict(v.strip().split(':',1) for v in s.split(';') if ':' in v)
 def path_points(d):
  tok=re.findall(r'[MLCQZmlcqz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',d);i=0;cur=np.array([0.,0.]);start=cur.copy();cmd=None;out=[]
  while i<len(tok):
   if tok[i].isalpha():cmd=tok[i];i+=1
   if cmd in 'Zz':out.append({'close':{}});cur=start.copy();cmd=None;continue
   count={'M':2,'L':2,'C':6,'Q':4}[cmd.upper()];v=np.array(list(map(float,tok[i:i+count])));i+=count
   if cmd.upper() in ['M','L']:
    q=v+(cur if cmd.islower() else 0);out.append({('moveTo' if cmd.upper()=='M' else 'lineTo'):{'x':float(q[0]),'y':float(q[1])}});cur=q
    if cmd.upper()=='M':start=q.copy();cmd='l' if cmd.islower() else 'L'
   else:
    pts=v.reshape(-1,2)+(cur if cmd.islower() else 0);p0=cur.copy()
    for u in np.linspace(0,1,13)[1:]:
     q=(1-u)**3*p0+3*(1-u)**2*u*pts[0]+3*(1-u)*u*u*pts[1]+u**3*pts[2] if cmd.upper()=='C' else (1-u)**2*p0+2*(1-u)*u*pts[0]+u*u*pts[1]
     out.append({'lineTo':{'x':float(q[0]),'y':float(q[1])}})
    cur=pts[-1]
  return out
 def walk(n,sty={}):
  st={**sty,**style(n.attrib.get('style',''))};tag=n.tag.split('}')[-1]
  if tag=='defs':return
  if tag in ['text','g'] and n.attrib.get('id','').startswith('text_'):return
  if tag in ['path','use']:
   el=n;ox=oy=0
   if tag=='use':
    href=n.attrib.get('{http://www.w3.org/1999/xlink}href','');el=defs.get(href[1:]);ox=float(n.attrib.get('x',0));oy=float(n.attrib.get('y',0))
   if el is not None and 'd' in el.attrib:
    ps=path_points(el.attrib['d']);ss={**style(el.attrib.get('style','')),**st}
    for q in ps:
     for k,v in q.items():
      if k!='close':v['x']+=ox;v['y']+=oy
    paths.append(dict(dash=ss.get('stroke-dasharray','none')!='none',commands=ps,fill=ss.get('fill','#000000'),stroke=ss.get('stroke','none'),width=float(ss.get('stroke-width',1)),opacity=float(ss.get('opacity',1)),fillopacity=float(ss.get('fill-opacity',1)),strokeopacity=float(ss.get('stroke-opacity',1))))
  for ch in n:walk(ch,st)
 walk(root)
 
 drawn=[]
 original=renderer.draw_text
 def capture(gc,x,y,ss,prop,angle,ismath=False,mtext=None):
  drawn.append((x,y,ss,prop.copy(),angle,ismath))
  return original(gc,x,y,ss,prop,angle,ismath=ismath,mtext=mtext)
 renderer.draw_text=capture
 fig.draw(renderer)
 renderer.draw_text=original
 for x,y,ss,prop,angle,ismath in drawn:
  tw,th,des=renderer.get_text_width_height_descent(ss,prop,ismath);factor=72/fig.dpi;tw*=factor;th*=factor;des*=factor;x*=factor;y*=factor
  r=np.radians(-angle);dx=tw/2;dy=-th/2+des;cx=x+np.cos(r)*dx-np.sin(r)*dy;cy=y+np.sin(r)*dx+np.cos(r)*dy
  width=tw+5;height=th+5
  texts.append(dict(text=ss,x=cx-width/2,y=cy-height/2,w=width,h=height,rotation=(-angle)%360,size=prop.get_size_in_points(),bold=prop.get_weight() in ['bold',700],color='#000000'))
 
 jsonpath.write_text(json.dumps(dict(width=W,height=H,texts=texts,paths=paths)))
