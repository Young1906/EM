all: dev pdf  clean_tex clean_src

dev:
	python src \
		--M 2 \
		--max_iter 100 \
		--tol 0.001 \
		--init kmean

pdf:
	cd tex; pdflatex main.tex; 


# ================================================== #
# Clean up
# ================================================== #
clean_tex:
	find tex -maxdepth 1 -type f \
		! -name "*.tex" \
		! -name "*.pdf" \
		! -name "*.bib" \
		! -name "*.png" \
		! -name "*.jpg" \
		-delete;
clean_src:
	rm -rf src/__pycache__;
