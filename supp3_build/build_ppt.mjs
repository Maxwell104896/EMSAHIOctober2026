import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {pathToFileURL} from 'node:url';
const root=process.env.SUPP3_WORKSPACE||new URL('..',import.meta.url).pathname.replace(/\/$/,''),K=96/72,skill='/root/.codex/skills/builtins/presentations';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
function col(c,op=1){if(c==='none')return 'none';if(!c||c==='currentColor')c='#000000';return op<1?`${c}/${Math.round(op*100)}`:c;}
for(const [stem,keys,caption]of [['Supplementary_Figures_S3_S4',['s3','s4'],null]]){
 const first=JSON.parse(await fs.readFile(root+'/supp3_build/'+keys[0]+'.json','utf8'));const p=Presentation.create({slideSize:{width:first.width*K,height:first.height*K}});let no=0;
 for(let key of keys){
  const D=JSON.parse(await fs.readFile(root+'/supp3_build/'+key+'.json','utf8'));let s=p.slides.add();s.background.fill='#FFFFFF';
  for(const o of D.paths){
   let xs=[],ys=[];for(let q of o.commands)for(let [k,v]of Object.entries(q))if(k!=='close'){xs.push(v.x);ys.push(v.y)}
   if(!xs.length)continue;let l=Math.min(...xs),t=Math.min(...ys),w=Math.max(Math.max(...xs)-l,.02),h=Math.max(Math.max(...ys)-t,.02);
   let commands=o.commands.map(q=>{let [k,v]=Object.entries(q)[0];return k==='close'?q:{[k]:{x:(v.x-l)*K,y:(v.y-t)*K}}});
   s.shapes.add({geometry:'custom',name:'Data mark '+(++no),position:{left:l*K,top:t*K,width:w*K,height:h*K},fill:col(o.fill,o.opacity*o.fillopacity),line:{style:o.dash?'dashed':'solid',fill:col(o.stroke,o.opacity*o.strokeopacity),width:o.stroke==='none'?0:o.width*K},customPaths:[{width:w*K,height:h*K,commands}]});
  }
  for(const o of D.texts){let sh=s.shapes.add({geometry:'textbox',name:'Editable label '+(++no),position:{left:o.x*K,top:o.y*K,width:o.w*K,height:o.h*K,rotation:o.rotation},fill:'none',line:{fill:'none',width:0}});sh.text=o.text;sh.text.style={typeface:'Times New Roman',fontSize:o.size*K,bold:o.bold,color:'#000000',alignment:'center',verticalAlignment:'middle',wrap:'none',autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};}
  s.speakerNotes.textFrame.setText(await fs.readFile(root+'/supp3_final/'+(key==='s3'?'S3_caption.txt':'S4_caption.txt'),'utf8'));
 }
 const candidate=root+'/supp3_build/'+stem+'_candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
 let r=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:root+'/supp3_final/'+stem+'.pptx',pythonExecutable:process.env.CODEX_PRIMARY_RUNTIME_PYTHON,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu',`${Math.round(first.width*12700)},${Math.round(first.height*12700)}`],explicitTotalSlideCount:keys.length,requiredNativeChartOwnerSlides:[],requiredNativeTableOwnerSlides:[],fontPolicy:{basis:'design',families:['Times New Roman']},verifyArtifactToolImport:true,receiptPath:root+'/supp3_build/'+stem+'_v3_validation.json'});console.log(r.finalPath,'objects',no);
 for(let j=0;j<keys.length;j++){const png=await p.export({slide:p.slides.items[j],format:'png',scale:2});await fs.writeFile(root+'/supp3_build/'+keys[j]+'_preview.png',new Uint8Array(await png.arrayBuffer()));}
}
