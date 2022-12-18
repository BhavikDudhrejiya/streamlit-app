
from bing_image_downloader import downloader
'''
query_string : String to be searched.
limit : (optional, default is 100) Number of images to download.
output_dir : (optional, default is 'dataset') Name of output dir.
adult_filter_off : (optional, default is True) Enable of disable adult filteration.
force_replace : (optional, default is False) Delete folder if present and start a fresh download.
timeout : (optional, default is 60) timeout for connection in seconds.
filter : (optional, default is "") filter, choose from [line, photo, clipart, gif, transparent]
verbose : (optional, default is True) Enable downloaded message.

'''
downloader.download('Random', 
                    limit=50,  
                    output_dir='/home/bhavik/Downloads', 
                    adult_filter_off=False, 
                    force_replace=False, 
                    timeout=60,
                    filter='photo',
                    verbose=True)
