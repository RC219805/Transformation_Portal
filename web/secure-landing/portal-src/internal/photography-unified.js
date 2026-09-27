import { buildPhotographyArgs, createPhotographyConfig, photographyConfigFromArgs } from "./photography.js";
import { buildPhotographyV6Args, createPhotographyV6Config, photographyV6ConfigFromArgs } from "./photography-v6.js";

// Each workflow retains its own draft. Runtime paths, cache authority, and
// execution engines are selected by the server, never imported from a profile.
export function createUnifiedPhotographyConfig(previous = {}) {
  previous = previous && typeof previous === "object" ? previous : {};
  return {
    workflow: previous.workflow === undefined ? "process" : previous.workflow,
    process: { ...createPhotographyV6Config(), ...previous.process },
    infer: { ...createPhotographyConfig(), ...previous.infer }
  };
}

export function isCompositePhotographyPipeline(pipeline, workflow = "process") {
  return pipeline === "lux-depth-v6" || (pipeline === "lux-depth" && workflow === "process");
}

export function buildUnifiedPhotographyArgs(config = {}, controls = {}) {
  const workflow = config.workflow === undefined ? "process" : config.workflow;
  if (workflow === "infer") return { workflow, ...buildPhotographyArgs(config.infer, controls.infer) };
  if (workflow === "process") return { workflow, ...buildPhotographyV6Args(config.process, controls.process) };
  // Preserve invalid imported workflow values for rejection by current preview.
  return { workflow };
}

export function unifiedPhotographyConfigFromArgs(args = {}, previous = {}) {
  const config = createUnifiedPhotographyConfig(previous);
  config.workflow = args.workflow === undefined ? "process" : args.workflow;
  if (config.workflow === "infer") config.infer = photographyConfigFromArgs(args, config.infer);
  if (config.workflow === "process") config.process = photographyV6ConfigFromArgs(args, config.process);
  return config;
}
