import { s as stateDiagram_default, S as StateDB, b as stateRenderer_v3_unified_default, a as styles_default } from "./chunk-4IRHCMPZ.B4kEjste.js";
import { _ as __name } from "./mermaid.core.CsvmAsas.js";
var diagram = {
  parser: stateDiagram_default,
  get db() {
    return new StateDB(2);
  },
  renderer: stateRenderer_v3_unified_default,
  styles: styles_default,
  init: /* @__PURE__ */ __name((cnf) => {
    if (!cnf.state) {
      cnf.state = {};
    }
    cnf.state.arrowMarkerAbsolute = cnf.arrowMarkerAbsolute;
  }, "init")
};
export {
  diagram
};
