from DS import DS
from datetime import datetime
g_path = r'C:\result-' + datetime.now().strftime('%Y-%m-%d-%H-%M')
DS.create_files(3,2, g_path)