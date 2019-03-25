## The code in this file is used for early-stage analysis

def orphadata():
    data_folder = "../data/orphadata/"
    hpo_file = data_folder + "en_product4_HPO.xml"

    import xml.etree.ElementTree as ET
    tree = ET.parse(hpo_file)
    root = tree.getroot()

if __name__ == '__main__':
    pass