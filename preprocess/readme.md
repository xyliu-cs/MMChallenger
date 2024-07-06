To preprocess OMCS text files,
you need to first download the [OMCS raw text file](https://github.com/commonsense/conceptnet5/wiki/Downloads#raw-sentences). 
```bash
wget https://s3.amazonaws.com/conceptnet/downloads/2018/omcs-sentences-free.txt
```
Then, run
`python omcs_preprocess.py path_to_downloaded_text output_dir` for preprocessing.