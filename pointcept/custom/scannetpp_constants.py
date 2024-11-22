"""
{'wall': 0, 'ceiling': 1, 'floor': 2, 'table': 3, 'door': 4, 'ceiling lamp': 5, 'cabinet': 6, 'blinds': 7, 'curtain': 8, 'chair': 9, 'storage cabinet': 10, 'office chair': 11, 'bookshelf': 12, 'whiteboard': 13, 'window': 14, 'box': 15, 'window frame': 16, 'monitor': 17, 'shelf': 18, 'doorframe': 19, 'pipe': 20, 'heater': 21, 'kitchen cabinet': 22, 'sofa': 23, 'windowsill': 24, 'bed': 25, 'shower wall': 26, 'trash can': 27, 'book': 28, 'plant': 29, 'blanket': 30, 'tv': 31, 'computer tower': 32, 'kitchen counter': 33, 'refrigerator': 34, 'jacket': 35, 'electrical duct': 36, 'sink': 37, 'bag': 38, 'picture': 39, 'pillow': 40, 'towel': 41, 'suitcase': 42, 'backpack': 43, 'crate': 44, 'keyboard': 45, 'rack': 46, 'toilet': 47, 'paper': 48, 'printer': 49, 'poster': 50, 'painting': 51, 'microwave': 52, 'board': 53, 'shoes': 54, 'socket': 55, 'bottle': 56, 'bucket': 57, 'cushion': 58, 'basket': 59, 'shoe rack': 60, 'telephone': 61, 'file folder': 62, 'cloth': 63, 'blind rail': 64, 'laptop': 65, 'plant pot': 66, 'exhaust fan': 67, 'cup': 68, 'coat hanger': 69, 'light switch': 70, 'speaker': 71, 'table lamp': 72, 'air vent': 73, 'clothes hanger': 74, 'kettle': 75, 'smoke detector': 76, 'container': 77, 'power strip': 78, 'slippers': 79, 'paper bag': 80, 'mouse': 81, 'cutting board': 82, 'toilet paper': 83, 'paper towel': 84, 'pot': 85, 'clock': 86, 'pan': 87, 'tap': 88, 'jar': 89, 'soap dispenser': 90, 'binder': 91, 'bowl': 92, 'tissue box': 93, 'whiteboard eraser': 94, 'toilet brush': 95, 'spray bottle': 96, 'headphones': 97, 'stapler': 98, 'marker': 99}
"""

# ScanNetpp Benchmark constants
# Semantic classes, 100
CLASS_LABELS_PP = (
    'wall', 
    'ceiling', 
    'floor', 
    'table', 
    'door', 
    'ceiling lamp', 
    'cabinet', 
    'blinds', 
    'curtain', 
    'chair', 
    'storage cabinet', 
    'office chair', 
    'bookshelf', 
    'whiteboard', 
    'window', 
    'box', 
    'window frame', 
    'monitor', 
    'shelf', 
    'doorframe', 
    'pipe', 
    'heater', 
    'kitchen cabinet', 
    'sofa', 
    'windowsill', 
    'bed', 
    'shower wall', 
    'trash can', 
    'book', 
    'plant', 
    'blanket', 
    'tv', 
    'computer tower', 
    'kitchen counter', 
    'refrigerator', 
    'jacket', 
    'electrical duct', 
    'sink', 
    'bag', 
    'picture', 
    'pillow', 
    'towel', 
    'suitcase', 
    'backpack', 
    'crate', 
    'keyboard', 
    'rack', 
    'toilet', 
    'paper', 
    'printer', 
    'poster', 
    'painting', 
    'microwave', 
    'board', 
    'shoes', 
    'socket', 
    'bottle', 
    'bucket', 
    'cushion', 
    'basket', 
    'shoe rack', 
    'telephone', 
    'file folder', 
    'cloth', 
    'blind rail', 
    'laptop', 
    'plant pot', 
    'exhaust fan', 
    'cup', 
    'coat hanger', 
    'light switch', 
    'speaker', 
    'table lamp', 
    'air vent', 
    'clothes hanger', 
    'kettle', 
    'smoke detector', 
    'container', 
    'power strip', 
    'slippers', 
    'paper bag', 
    'mouse', 
    'cutting board', 
    'toilet paper', 
    'paper towel', 
    'pot', 
    'clock', 
    'pan', 
    'tap', 
    'jar', 
    'soap dispenser', 
    'binder', 
    'bowl', 
    'tissue box', 
    'whiteboard eraser', 
    'toilet brush', 
    'spray bottle', 
    'headphones', 
    'stapler', 
    'marker',
)

# Instance classes, 84
INST_LABELS_PP = ( 
    'table', 
    'door', 
    'ceiling lamp', 
    'cabinet', 
    'blinds', 
    'curtain', 
    'chair', 
    'storage cabinet', 
    'office chair', 
    'bookshelf', 
    'whiteboard', 
    'window', 
    'box', 
    'monitor', 
    'shelf', 
    'heater', 
    'kitchen cabinet', 
    'sofa', 
    'bed', 
    'trash can', 
    'book', 
    'plant', 
    'blanket', 
    'tv', 
    'computer tower', 
    'refrigerator', 
    'jacket', 
    'sink', 
    'bag', 
    'picture', 
    'pillow', 
    'towel', 
    'suitcase', 
    'backpack', 
    'crate', 
    'keyboard', 
    'rack', 
    'toilet', 
    'printer', 
    'poster', 
    'painting', 
    'microwave', 
    'shoes', 
    'socket', 
    'bottle', 
    'bucket', 
    'cushion', 
    'basket', 
    'shoe rack', 
    'telephone', 
    'file folder', 
    'laptop', 
    'plant pot', 
    'exhaust fan', 
    'cup', 
    'coat hanger', 
    'light switch', 
    'speaker', 
    'table lamp', 
    'kettle', 
    'smoke detector', 
    'container', 
    'power strip', 
    'slippers', 
    'paper bag', 
    'mouse', 
    'cutting board', 
    'toilet paper', 
    'paper towel', 
    'pot', 
    'clock', 
    'pan', 
    'tap', 
    'jar', 
    'soap dispenser', 
    'binder', 
    'bowl', 
    'tissue box', 
    'whiteboard eraser', 
    'toilet brush', 
    'spray bottle', 
    'headphones', 
    'stapler', 
    'marker',
)