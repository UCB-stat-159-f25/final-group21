import kagglehub

def get_full_data():
	path = kagglehub.dataset_download("usdot/flight-delays")
	
	return path


