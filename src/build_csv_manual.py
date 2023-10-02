import os
from grammar_utilities import build_csv

script_directory = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(script_directory, '..', 'data', 'cfg-output')

# 2.1 from Lan et. al
cfg = {
    'S': [['S_FG'], ['S_XG'], ['S_FX'], ['S_XX']],
    'S_FG': [['PREAMBLE', 'F', 'G']],
    'S_XG': [['UNGRAMMATICAL', 'PREAMBLE', 'XF', 'G']],
    'S_FX': [['UNGRAMMATICAL', 'PREAMBLE', 'F', 'XG']],
    'S_XX': [['PREAMBLE', 'XF', 'XG']],
    'UNGRAMMATICAL': [['*']],
    'NAME1': [['Michael'], ['Ashley'], ['Daniel'], ['John'], ['Brandon'], ['William'], ['Nicole'], ['Eric'], ['Melissa'], ['Timothy']],
    'NAME2': [['Christopher'], ['Jennifer'], ['David']],
    'NAME3': [['Jessica'], ['Joshua'], ['James']],
    'NAME4': [['Matthew'], ['you']],
    'PREAMBLE': [['I know']],
    'F': [['what', 'NAME1', 'V1']],
    'XF': [['that', 'NAME1', 'V1', 'OBJ1']],
    'CONN': [['yesterday and will']],
    'G': [['CONN', 'V2', 'ADJUNCT']],
    'XG': [['CONN', 'V2', 'OBJ2', 'ADJUNCT']],
    'V1': [['looked for'], ['searched everywhere for'], ['found'], ['bought'], ['purchased'], ['went shopping for']],
    'OBJ1': [['food'], ['bread'], ['meat'], ['cheese'], ['candy']],
    'V2': [['devour'], ['serve'], ['donate'], ['distribute']],
    'OBJ2': [['it'], ['fish'], ['snacks']],
    'ADJUNCT': [['tomorrow'], ['soon'], ['tonight'], ['today'], ['shortly'], ['quickly']],
}

output_file_path = os.path.join(
    output_directory, 'manual_output.csv')

build_csv(cfg, ['S_FG', 'S_XG', 'S_FX', 'S_XX'], output_file_path)
