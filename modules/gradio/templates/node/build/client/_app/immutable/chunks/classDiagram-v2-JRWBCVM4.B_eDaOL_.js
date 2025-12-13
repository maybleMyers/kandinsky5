import { c as classDiagram_default, C as ClassDB, a as classRenderer_v3_unified_default, s as styles_default } from "./chunk-5V4FS25O.CE6I_miO.js";
import { _ as __name } from "./mermaid.core.CsvmAsas.js";
var diagram = {
  parser: classDiagram_default,
  get db() {
    return new ClassDB();
  },
  renderer: classRenderer_v3_unified_default,
  styles: styles_default,
  init: /* @__PURE__ */ __name((cnf) => {
    if (!cnf.class) {
      cnf.class = {};
    }
    cnf.class.arrowMarkerAbsolute = cnf.arrowMarkerAbsolute;
  }, "init")
};
export {
  diagram
};
