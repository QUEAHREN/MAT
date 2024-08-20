pip install scikit-image wandb calflops transformers

python basicsr/test.py -opt options/test/MAT_GoPro.yml
python basicsr/test.py -opt options/test/MAT_HSERGB.yml
python basicsr/test.py -opt options/test/MAT_REBlur.yml
python basicsr/test.py -opt options/test/MAT_REVD.yml
