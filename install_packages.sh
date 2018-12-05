# For Mac and Linux
while read requirement; do conda install --yes $requirement; done < requirements.txt

# For Windows
#FOR /F "delims=~" %f in (requirements.txt) DO conda install --yes "%f" || pip install "%f"
