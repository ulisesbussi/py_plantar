import numpy as np
import json
from glob import glob


def load_and_parse_json(json_file):
    keys_to_parse = ['accelerometerX', 'accelerometerY', 'accelerometerZ',
                     'coord_lat', 'coord_lng',
                     'magnetometerX', 'magnetometerY', 'magnetometerZ'
                    ]
    with open(json_file) as f:
        data = json.load(f)
    for key in keys_to_parse:
        try:
            data[key] = float(data[key])
        except ValueError:
            print(f"{key} is not a float")
        except KeyError:
            print(f"{key} is not in the json file")
    return data

def calc_angles(accel: np.ndarray,
                mag: np.ndarray) -> np.ndarray:
    """Calcula los angulos como yaw, pitch, roll a partir de las 
    mediciones de acelerometro y magnetometro

    Args:
        accel (np.ndarray): vector de medicion de acelerometro 
            arrojado por android como 
            [accelerometerX, accelerometerY, accelerometerZ]
        mag (np.ndarray): vector de medicion de magnetometro
            arrojado por android como
            [magnetometerX, magnetometerY, magnetometerZ]

    Returns:
        np.ndarray: angulos de rotacion en radianes como 
            [yaw, pitch, roll]
    """
    h = np.cross(mag,accel) #primero mag

    h_norm = h/np.linalg.norm(h) 
    a_norm = accel/np.linalg.norm(accel)
    m_norm = mag/np.linalg.norm(mag) #este no lo uso

    m2 = np.cross(a_norm,h_norm)
    rot_mat = np.array([h_norm,m2,a_norm])

    v = np.array( [ np.arctan2(rot_mat[0,1],rot_mat[1,1]),
                    np.arcsin(rot_mat[2,1]),
                    np.arctan2(-rot_mat[2,0],rot_mat[2,2])
                ])
    return v    

def read_img_and_metadata(img_path: str) -> tuple:
    """Lee la imagen y la metadata de la imagen

    Args:
        img_path (str): ruta de la imagen

    Returns:
        tuple: imagen y metadata
    """
    img = cv2.imread(img_path)
    metadata = load_and_parse_json('/'.join(img_path.split('.')[:-1]) + '.json')
    
    return img, metadata

def get_mag_accel_gps(metadata:dict) -> tuple:
    """Obtiene los vectores de magnetometro, acelerometro y gps

    Args:
        metadata (dict): metadata de la imagen

    Returns:
        tuple: vectores de magnetometro, acelerometro y gps (lon,lat)
    """
    mag = np.array([metadata['magnetometerX'],
                    metadata['magnetometerY'],
                    metadata['magnetometerZ']])
    accel = np.array([metadata['accelerometerX'],
                      metadata['accelerometerY'],
                      metadata['accelerometerZ']])
    gps = np.array([metadata['coord_lng'],
                    metadata['coord_lat']])
    return mag, accel, gps
    
def points_coords_from_folder(folder_path : str, 
                              output_filename : str,
                              src: str = 'epsg:4326',
                              verbose: str = True) -> None:
    """crea un .csv con los datos de las coordenadas de los puntos 
    donde se sacaron las imagenes para importar a qgis. Lee los datos
    de los .json de la carpeta folder_path, transforma al sistema de 
    referencia src (si es distinto de 'epsg:4326') y guarda los puntos 
    en un archivo .csv con las columnas 'nombre', x (lng), y (lat).

    Args:
        folder_path (str): folder from where to read the .json files
        output_filename (str): name of the output .csv file
        src (str, optional): source system of reference. Defaults to 'epsg:4326'.
        verboce (bool, optional): print the progress. Defaults to True.
    """
    from glob import glob
    import pandas as pd
    from pyproj import CRS, Transformer
    from tqdm import tqdm

    if folder_path[-1] != '/':
        folder_path += '/'
    files = glob(f'{folder_path}*.json')

    points = []
    if verbose: print('Reading files...')
    iter = tqdm(files) if verbose else files
    for f in iter:
        name = f.split('/')[-1].split('.')[0]
        data = load_and_parse_json(f)
        point = [name, data['coord_lng'], data['coord_lat']]
        points.append(point)
        
 
    df = pd.DataFrame(points, columns=['nombre', 'x', 'y'])
    
    
    
    if src != 'epsg:4326':
        transformer = Transformer.from_crs('epsg:4326', src)
        df[['x', 'y']] = df[['x', 'y']].apply(lambda x: transformer.transform(x[0], x[1]), axis=1)
    if verbose: print('Saving file...')
    df.to_csv(output_filename, index=False)
    if verbose: print('Done!')
 