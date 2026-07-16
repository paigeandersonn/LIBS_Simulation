// MathJax v3 configuration for MkDocs Material + pymdownx.arithmatex (generic mode).
// arithmatex emits math wrapped in `.arithmatex` elements using \( \) / \[ \]
// delimiters, so MathJax is told to typeset only that class. The document$
// subscription re-typesets after Material's instant (client-side) navigation,
// which otherwise swaps page content without a full reload.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
