# NPSV: Non-parametric Structural Variant Genotyper

```
docker run --entrypoint /bin/bash --shm-size=6g -v `pwd`:/opt/npsv2 -v ~/Research/Data:/data -w /opt/npsv2 -it npsv2
```

Generate examples with simulated replicates:
```
npsv2 examples \
    -r /data/human_g1k_v37.fasta \
    -i tests/data/1_899922_899992_DEL.vcf.gz \
    -b tests/data/1_896922_902998.bam \
    -o tests/results/test.tfresults \
    --read-length 148 \
    --fragment-mean 573 \
    --fragment-sd 164 \
    --depth 25 \
    --replicates 2
```

Visualize examples (showing up to two replicates in the image):
```
npsv2 visualize \
    -i tests/results/test.tfresults \
    -o tests/results \
    --replicates 2
```