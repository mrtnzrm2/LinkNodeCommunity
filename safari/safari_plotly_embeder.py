import chart_studio.tools as tls


username = 'jmarti53' # your username
api_key = 'Z44nwG##g_nk7k8' # your api key - go to profile > settings > regenerate key
tls.set_credentials_file(username=username, api_key=api_key)

url = "https://ftp.jorgemartinezarmas.com/Obsidian/LINKPROJECT/fMRI/product-positive.html"

e = tls.get_embed(url, file_id=12345)
print(e)