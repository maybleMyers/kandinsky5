import { p as populateCommonDb } from "./chunk-IUKPXING.BdwJKnSN.js";
import { E as defaultConfig_default, o as setDiagramTitle, p as getDiagramTitle, s as setAccTitle, g as getAccTitle, c as setAccDescription, b as getAccDescription, _ as __name, l as log, v as clear, d as getConfig2$1, F as cleanAndMerge, K as selectSvgElement, Q as parseFontSize, k as configureSvgSize } from "./mermaid.core.CsvmAsas.js";
import { p as parse } from "./mermaid-parser.core.B39mVJqj.js";
import { d as d3arc } from "./arc.Mz__KNAe.js";
import { o as ordinal } from "./ordinal.Bkur82De.js";
import { a as constant, t as tau, n as array } from "./step.D3xxTXJ6.js";
function descending(a, b) {
  return b < a ? -1 : b > a ? 1 : b >= a ? 0 : NaN;
}
function identity(d) {
  return d;
}
function d3pie() {
  var value = identity, sortValues = descending, sort = null, startAngle = constant(0), endAngle = constant(tau), padAngle = constant(0);
  function pie(data) {
    var i, n = (data = array(data)).length, j, k, sum = 0, index = new Array(n), arcs = new Array(n), a0 = +startAngle.apply(this, arguments), da = Math.min(tau, Math.max(-tau, endAngle.apply(this, arguments) - a0)), a1, p = Math.min(Math.abs(da) / n, padAngle.apply(this, arguments)), pa = p * (da < 0 ? -1 : 1), v;
    for (i = 0; i < n; ++i) {
      if ((v = arcs[index[i] = i] = +value(data[i], i, data)) > 0) {
        sum += v;
      }
    }
    if (sortValues != null)
      index.sort(function(i2, j2) {
        return sortValues(arcs[i2], arcs[j2]);
      });
    else if (sort != null)
      index.sort(function(i2, j2) {
        return sort(data[i2], data[j2]);
      });
    for (i = 0, k = sum ? (da - n * pa) / sum : 0; i < n; ++i, a0 = a1) {
      j = index[i], v = arcs[j], a1 = a0 + (v > 0 ? v * k : 0) + pa, arcs[j] = {
        data: data[j],
        index: i,
        value: v,
        startAngle: a0,
        endAngle: a1,
        padAngle: p
      };
    }
    return arcs;
  }
  pie.value = function(_) {
    return arguments.length ? (value = typeof _ === "function" ? _ : constant(+_), pie) : value;
  };
  pie.sortValues = function(_) {
    return arguments.length ? (sortValues = _, sort = null, pie) : sortValues;
  };
  pie.sort = function(_) {
    return arguments.length ? (sort = _, sortValues = null, pie) : sort;
  };
  pie.startAngle = function(_) {
    return arguments.length ? (startAngle = typeof _ === "function" ? _ : constant(+_), pie) : startAngle;
  };
  pie.endAngle = function(_) {
    return arguments.length ? (endAngle = typeof _ === "function" ? _ : constant(+_), pie) : endAngle;
  };
  pie.padAngle = function(_) {
    return arguments.length ? (padAngle = typeof _ === "function" ? _ : constant(+_), pie) : padAngle;
  };
  return pie;
}
var DEFAULT_PIE_CONFIG = defaultConfig_default.pie;
var DEFAULT_PIE_DB = {
  sections: /* @__PURE__ */ new Map(),
  showData: false,
  config: DEFAULT_PIE_CONFIG
};
var sections = DEFAULT_PIE_DB.sections;
var showData = DEFAULT_PIE_DB.showData;
var config = structuredClone(DEFAULT_PIE_CONFIG);
var getConfig2 = /* @__PURE__ */ __name(() => structuredClone(config), "getConfig");
var clear2 = /* @__PURE__ */ __name(() => {
  sections = /* @__PURE__ */ new Map();
  showData = DEFAULT_PIE_DB.showData;
  clear();
}, "clear");
var addSection = /* @__PURE__ */ __name(({ label, value }) => {
  if (!sections.has(label)) {
    sections.set(label, value);
    log.debug(`added new section: ${label}, with value: ${value}`);
  }
}, "addSection");
var getSections = /* @__PURE__ */ __name(() => sections, "getSections");
var setShowData = /* @__PURE__ */ __name((toggle) => {
  showData = toggle;
}, "setShowData");
var getShowData = /* @__PURE__ */ __name(() => showData, "getShowData");
var db = {
  getConfig: getConfig2,
  clear: clear2,
  setDiagramTitle,
  getDiagramTitle,
  setAccTitle,
  getAccTitle,
  setAccDescription,
  getAccDescription,
  addSection,
  getSections,
  setShowData,
  getShowData
};
var populateDb = /* @__PURE__ */ __name((ast, db2) => {
  populateCommonDb(ast, db2);
  db2.setShowData(ast.showData);
  ast.sections.map(db2.addSection);
}, "populateDb");
var parser = {
  parse: /* @__PURE__ */ __name(async (input) => {
    const ast = await parse("pie", input);
    log.debug(ast);
    populateDb(ast, db);
  }, "parse")
};
var getStyles = /* @__PURE__ */ __name((options) => `
  .pieCircle{
    stroke: ${options.pieStrokeColor};
    stroke-width : ${options.pieStrokeWidth};
    opacity : ${options.pieOpacity};
  }
  .pieOuterCircle{
    stroke: ${options.pieOuterStrokeColor};
    stroke-width: ${options.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${options.pieTitleTextSize};
    fill: ${options.pieTitleTextColor};
    font-family: ${options.fontFamily};
  }
  .slice {
    font-family: ${options.fontFamily};
    fill: ${options.pieSectionTextColor};
    font-size:${options.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${options.pieLegendTextColor};
    font-family: ${options.fontFamily};
    font-size: ${options.pieLegendTextSize};
  }
`, "getStyles");
var pieStyles_default = getStyles;
var createPieArcs = /* @__PURE__ */ __name((sections2) => {
  const pieData = [...sections2.entries()].map((element) => {
    return {
      label: element[0],
      value: element[1]
    };
  }).sort((a, b) => {
    return b.value - a.value;
  });
  const pie = d3pie().value(
    (d3Section) => d3Section.value
  );
  return pie(pieData);
}, "createPieArcs");
var draw = /* @__PURE__ */ __name((text, id, _version, diagObj) => {
  log.debug("rendering pie chart\n" + text);
  const db2 = diagObj.db;
  const globalConfig = getConfig2$1();
  const pieConfig = cleanAndMerge(db2.getConfig(), globalConfig.pie);
  const MARGIN = 40;
  const LEGEND_RECT_SIZE = 18;
  const LEGEND_SPACING = 4;
  const height = 450;
  const pieWidth = height;
  const svg = selectSvgElement(id);
  const group = svg.append("g");
  group.attr("transform", "translate(" + pieWidth / 2 + "," + height / 2 + ")");
  const { themeVariables } = globalConfig;
  let [outerStrokeWidth] = parseFontSize(themeVariables.pieOuterStrokeWidth);
  outerStrokeWidth ?? (outerStrokeWidth = 2);
  const textPosition = pieConfig.textPosition;
  const radius = Math.min(pieWidth, height) / 2 - MARGIN;
  const arcGenerator = d3arc().innerRadius(0).outerRadius(radius);
  const labelArcGenerator = d3arc().innerRadius(radius * textPosition).outerRadius(radius * textPosition);
  group.append("circle").attr("cx", 0).attr("cy", 0).attr("r", radius + outerStrokeWidth / 2).attr("class", "pieOuterCircle");
  const sections2 = db2.getSections();
  const arcs = createPieArcs(sections2);
  const myGeneratedColors = [
    themeVariables.pie1,
    themeVariables.pie2,
    themeVariables.pie3,
    themeVariables.pie4,
    themeVariables.pie5,
    themeVariables.pie6,
    themeVariables.pie7,
    themeVariables.pie8,
    themeVariables.pie9,
    themeVariables.pie10,
    themeVariables.pie11,
    themeVariables.pie12
  ];
  const color = ordinal(myGeneratedColors);
  group.selectAll("mySlices").data(arcs).enter().append("path").attr("d", arcGenerator).attr("fill", (datum) => {
    return color(datum.data.label);
  }).attr("class", "pieCircle");
  let sum = 0;
  sections2.forEach((section) => {
    sum += section;
  });
  group.selectAll("mySlices").data(arcs).enter().append("text").text((datum) => {
    return (datum.data.value / sum * 100).toFixed(0) + "%";
  }).attr("transform", (datum) => {
    return "translate(" + labelArcGenerator.centroid(datum) + ")";
  }).style("text-anchor", "middle").attr("class", "slice");
  group.append("text").text(db2.getDiagramTitle()).attr("x", 0).attr("y", -(height - 50) / 2).attr("class", "pieTitleText");
  const legend = group.selectAll(".legend").data(color.domain()).enter().append("g").attr("class", "legend").attr("transform", (_datum, index) => {
    const height2 = LEGEND_RECT_SIZE + LEGEND_SPACING;
    const offset = height2 * color.domain().length / 2;
    const horizontal = 12 * LEGEND_RECT_SIZE;
    const vertical = index * height2 - offset;
    return "translate(" + horizontal + "," + vertical + ")";
  });
  legend.append("rect").attr("width", LEGEND_RECT_SIZE).attr("height", LEGEND_RECT_SIZE).style("fill", color).style("stroke", color);
  legend.data(arcs).append("text").attr("x", LEGEND_RECT_SIZE + LEGEND_SPACING).attr("y", LEGEND_RECT_SIZE - LEGEND_SPACING).text((datum) => {
    const { label, value } = datum.data;
    if (db2.getShowData()) {
      return `${label} [${value}]`;
    }
    return label;
  });
  const longestTextWidth = Math.max(
    ...legend.selectAll("text").nodes().map((node) => (node == null ? void 0 : node.getBoundingClientRect().width) ?? 0)
  );
  const totalWidth = pieWidth + MARGIN + LEGEND_RECT_SIZE + LEGEND_SPACING + longestTextWidth;
  svg.attr("viewBox", `0 0 ${totalWidth} ${height}`);
  configureSvgSize(svg, height, totalWidth, pieConfig.useMaxWidth);
}, "draw");
var renderer = { draw };
var diagram = {
  parser,
  db,
  renderer,
  styles: pieStyles_default
};
export {
  diagram
};
