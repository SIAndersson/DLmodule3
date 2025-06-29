report: tex/main.pdf

clean:
	rm tex/main.aux tex/main.log tex/main.out tex/main.pdf

%.pdf: %.tex
	pdflatex -output-directory=$(@D) $^

.PHONY: report clean
