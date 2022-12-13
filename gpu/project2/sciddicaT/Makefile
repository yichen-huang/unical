# definisce la macro CPPC
ifndef CPPC
	CPPC=g++
endif

ifdef TEST
	HDR=../data/test_header.txt
	DEM=../data/test_dem.txt
	SRC=../data/test_source.txt
	OUT=./test_output_OpenMP
	OUT_SERIAL=./test_output_serial
	STEPS=2
else
	HDR=../data/tessina_header.txt
	DEM=../data/tessina_dem.txt
	SRC=../data/tessina_source.txt
	OUT=./tessina_output_OpenMP
	OUT_SERIAL=./tessina_output_serial
	STEPS=4000
endif

# definisce le macro contenenti i nomei degli eseguibili
# e il numero di thread omp per la versione parallela
NT = 2 # numero di threads OpenMP
EXEC = sciddicaTomp
EXEC_SERIAL = sciddicaTserial

# definisce il target di default, utile in
# caso di invocazione di make senza parametri
default:all

# compila le versioni seriale e OpenMP
all:
	$(CPPC) sciddicaT.cpp -o $(EXEC) -fopenmp -O3
	$(CPPC) sciddicaT.cpp -o $(EXEC_SERIAL) -O3

# esegue la simulazione OpenMP
run_omp:
	OMP_NUM_THREADS=$(NT) ./$(EXEC) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) &&  md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

# esegue la simulazione seriale 
run:
	./$(EXEC_SERIAL) $(HDR) $(DEM) $(SRC) $(OUT_SERIAL) $(STEPS) &&  md5sum $(OUT_SERIAL) && cat $(HDR) $(OUT_SERIAL) > $(OUT_SERIAL).qgis && rm $(OUT_SERIAL)

# elimina l'eseguibile, file oggetto e file di output
clean:
	rm -f $(EXEC) $(EXEC_SERIAL) *.o *output*

# elimina file oggetto e di output
wipe:
	rm -f *.o *output*
