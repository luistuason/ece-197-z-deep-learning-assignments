import os
import requests
import tarfile

DATASET_URL = 'https://github.com/luistuason/object-detection/releases/download/v1.0.0/drinks.tar.gz'
TRAINED_MODEL_URL = 'https://github.com/luistuason/object-detection/releases/download/v1.0.0/trained_model.pth'


def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        print('-----------------------------------------------------')
        print('Drinks dataset not found.')
        print('Preparing to download drinks dataset...')
        url = DATASET_URL
            
        
        # Check if /data folder exists
        if not os.path.exists('./data'):
            print('Creating data folder')
            os.makedirs('./data')

        try:
            tar_path = './data/drinks.tar.gz'
            # Download
            req = requests.get(url, stream=True)
            if req.ok:
                print('Downloading drinks dataset...')
                with open(tar_path, 'wb') as f:
                    for chunk in req.iter_content(chunk_size=1024*8): 
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
                
                # Extract
                with tarfile.open(tar_path, 'r:gz') as f:
                    print('Extracting drinks dataset...')
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(f, "./data")

                os.remove(tar_path)
                print('Drinks dataset downloaded and extracted. Deleting tar file')
                print('----------------------------------------------------------')

            else:
                print('Download failed -- ', req.status_code, req.text)
                os.rmdir('./data')

        except Exception as e:
            print('Error downloading or extracting drinks dataset -- ', e)
            os.rmdir('./data')
    else:
        print('Drinks dataset found.')
        print('----------------------------------------------------------')

def load_model_checkpoint(trained_model_path):
    if not os.path.exists(trained_model_path):
        print('-----------------------------------------------------')
        print('Trained model checkpoint not found.')
        print('Preparing to download trained model checkpoint...')
        url = TRAINED_MODEL_URL

        # Check if /export folder exists
        if not os.path.exists('./export'):
            print('Creating export folder')
            os.makedirs('./export')

        try:
            req = requests.get(url, stream=True)
            if req.ok:
                print('Downloading trained model checkpoint file...')
                with open(trained_model_path, 'wb') as f:
                    for chunk in req.iter_content(chunk_size=1024*8): 
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
                print('Trained model checkpoint downloaded.')
                print('----------------------------------------------------------')
            else:
                print('Download failed -- ', req.status_code, req.text)
                os.rmdir('./export')

        except Exception as e:
            print('Error downloading trained model checkpoint -- ', e)
    else:
        print('Trained model checkpoint found.')
        print('----------------------------------------------------------')


