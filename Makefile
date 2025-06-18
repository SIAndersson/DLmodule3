report: tex/main.pdf

%.pdf: %.tex
	pdflatex -output-directory=$(@D) $^

.PHONY: report
