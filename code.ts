// code.ts — Presence Overlay Guides (quick plugin)
// Creates guides + locale prompts on current selection (frame) or a new frame.
// Default locale: US_EN. Change LOCALE to one of: US_EN, JP_JA, DE_DE, CN_ZH, IN_EN, GCC_AR.

const LOCALE = "US_EN";

const PROMPTS: Record<string,string[]> = {
  "US_EN": ["Silent yes","What would you do?","Stay with me"],
  "JP_JA": ["Consider carefully","Small nod","Soft gaze"],
  "DE_DE": ["Confirm plan","Focus point","Steady breath"],
  "CN_ZH": ["Thought resolved","Subtle assent","Calm gaze"],
  "IN_EN": ["Assure client","Yes, I have it","Warm gaze"],
  "GCC_AR":["Decision held","Assurance","Poised gaze"],
};

function ensureFrame(aspect: "4:5"|"2:3"): FrameNode {
  let frame: FrameNode | undefined = figma.currentPage.selection.find(n => n.type === "FRAME") as FrameNode;
  if (!frame) {
    frame = figma.createFrame();
    frame.name = `Presence Frame ${aspect}`;
    if (aspect === "4:5") { frame.resize(2400, 3000); }
    else { frame.resize(1200, 1800); }
    figma.currentPage.appendChild(frame);
  }
  return frame;
}

function addGuides(aspect: "4:5"|"2:3", locale: string) {
  const frame = ensureFrame(aspect);
  const W = frame.width, H = frame.height;

  // Container group
  const groupNodes: SceneNode[] = [];

  // Border
  const border = figma.createRectangle();
  border.resize(W, H);
  border.name = "Presence Border";
  border.strokes = [{type:"SOLID", color:{r:1,g:1,b:1}}];
  border.strokeWeight = 2;
  border.fills = [];
  groupNodes.push(border);

  // Eye line
  const eyePct = (aspect === "4:5") ? 0.27 : 0.36;
  const eye = figma.createRectangle();
  eye.resize(W-80, 4);
  eye.x = 40; eye.y = H*eyePct;
  eye.name = `Eye line ${Math.round(eyePct*100)}%`;
  eye.fills = [{type:"SOLID", color:{r:0.3137,g:0.7843,b:0.4706}, opacity:0.9}];
  eye.strokes = [];
  groupNodes.push(eye);

  // Gutters
  const gutter = 0.14;
  const left = figma.createRectangle(); left.resize(W*gutter, H);
  left.x = 0; left.y = 0; left.opacity = 0.12; left.fills = [{type:"SOLID", color:{r:1,g:1,b:1}, opacity:0.12}]; left.name = "Left gutter 14%";
  const right = figma.createRectangle(); right.resize(W*gutter, H);
  right.x = W - W*gutter; right.y = 0; right.opacity = 0.12; right.fills = [{type:"SOLID", color:{r:1,g:1,b:1}, opacity:0.12}]; right.name = "Right gutter 14%";
  groupNodes.push(left, right);

  // Brand band (top)
  const brand = figma.createRectangle();
  brand.resize(W, 2);
  brand.x = 0; brand.y = H*0.12;
  brand.name = "Brand band ~12%";
  brand.fills = [{type:"SOLID", color:{r:1,g:1,b:1}, opacity:0.5}];
  brand.strokes = [];
  groupNodes.push(brand);

  // Chest/Hands band
  const b1 = (aspect === "4:5") ? 0.68 : 0.78;
  const b2 = (aspect === "4:5") ? 0.80 : 0.95;
  const chest = figma.createRectangle();
  chest.resize(W, (H*b2 - H*b1));
  chest.x = 0; chest.y = H*b1;
  chest.name = "Chest/Hands band";
  chest.fills = [{type:"SOLID", color:{r:1,g:1,b:1}, opacity:0.10}];
  chest.strokes = [{type:"SOLID", color:{r:1,g:1,b:1}, opacity:0.35}];
  chest.strokeWeight = 1;
  groupNodes.push(chest);

  // Prompt text
  const prompts = PROMPTS[locale] || PROMPTS["US_EN"];
  const txt = figma.createText();
  txt.name = `Expression prompts (${locale})`;
  txt.characters = "Expression prompts:\n- " + prompts.join("\n- ");
  txt.x = 64; txt.y = 64;
  txt.fontSize = 28;
  groupNodes.push(txt);

  // Group and attach to frame
  const group = figma.group(groupNodes, frame);
  group.name = `Presence Guides ${aspect} (${locale})`;
  frame.appendChild(group);

  // Place inside the frame bounds
  group.x = 0; group.y = 0;
  figma.currentPage.selection = [group];
  figma.notify("Presence guides added");
}

figma.on("run", ({command}) => {
  const locale = LOCALE;
  if (command === "add-guides-2x3") addGuides("2:3", locale);
  else addGuides("4:5", locale);
  figma.closePlugin();
});
