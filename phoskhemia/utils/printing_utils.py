from itertools import cycle
import time
import os

decreasing_char_dens = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

def bouncing_loader(
        width: int | str='auto', 
        symbol: list[str] | str='o', 
        ends: str='|'
        ) -> None:

    if width == 'auto':
        width, _ = os.get_terminal_size()
        width: int = (
            width - 2 * len(ends) - len(symbol) if isinstance(symbol, str) 
            else (width - 2 * len(ends) - max([len(i) for i in symbol]) 
            if isinstance(symbol, list) else width - 3)
            )

    if isinstance(symbol, str):
        loading_symbols: list[str] = [
            f"{ends}{'' :<{i}}{symbol}{'' :<{width - i}}{ends}" 
            for i in range(width + 1)
            ]

    elif isinstance(symbol, list):
        symbol_width = max([len(i) for i in symbol])
        loading_symbols = [
            f"{ends}{'' :<{i}}"
            f"{f'{symbol[i % len(symbol)]}' :^{symbol_width}}"
            f"{'' :<{width - i}}{ends}" for i in range(width + 1)
            ]

    loading_symbols.extend(reversed(loading_symbols[1:-1]))

    for i, symbol in enumerate(cycle(loading_symbols)):
        print(f'{symbol}', end="\r")
        time.sleep(1 / 15)
        if i >= 150:
            break

#bouncing_loader(width='auto', symbol=["ᵒ", "o", "ₒ"])
#bouncing_loader(width='auto', symbol=['◜', '◝', '◞', '◟'])
inchworm = ['▟▀▙', '▄▄▄']
#bouncing_loader(width='auto', symbol=inchworm)

wave = ['▔▁']
#inchworm = ['◞⏜◟', '◞◠◟']

# ANSI control codes for direct terminal control. \033[ is the format used for 
# the control sequence introducer (CSI) commands ESC [
nlines = 9
#print(f"\033[{nlines}S", end="")
#print(f"\033[{nlines}A", end="")
#print(f"\033[s", end="")

loading_symbols = [
    ['o        ', ' o       ', '  o      ', '   o     ', '    o    ', '     o   ', '      o  ', '       o ', '        o'],
    [' o       ', '  o      ', '   o     ', '    o    ', '     o   ', '      o  ', '       o ', '        o', '       o '],
    ['  o      ', '   o     ', '    o    ', '     o   ', '      o  ', '       o ', '        o', '       o ', '      o  '],
    ['   o     ', '    o    ', '     o   ', '      o  ', '       o ', '        o', '       o ', '      o  ', '     o   '],
    ['    o    ', '     o   ', '      o  ', '       o ', '        o', '       o ', '      o  ', '     o   ', '    o    '],
    ['     o   ', '      o  ', '       o ', '        o', '       o ', '      o  ', '     o   ', '    o    ', '   o     '],
    ['      o  ', '       o ', '        o', '       o ', '      o  ', '     o   ', '    o    ', '   o     ', '  o      '],
    ['       o ', '        o', '       o ', '      o  ', '     o   ', '    o    ', '   o     ', '  o      ', ' o       '],
    ['        o', '       o ', '      o  ', '     o   ', '    o    ', '   o     ', '  o      ', ' o       ', 'o        '],
    ['       o ', '      o  ', '     o   ', '    o    ', '   o     ', '  o      ', ' o       ', 'o        ', ' o       '],
    ['      o  ', '     o   ', '    o    ', '   o     ', '  o      ', ' o       ', 'o        ', ' o       ', '  o      '],
    ['     o   ', '    o    ', '   o     ', '  o      ', ' o       ', 'o        ', ' o       ', '  o      ', '   o     '],
    ['    o    ', '   o     ', '  o      ', ' o       ', 'o        ', ' o       ', '  o      ', '   o     ', '    o    '],
    ['   o     ', '  o      ', ' o       ', 'o        ', ' o       ', '  o      ', '   o     ', '    o    ', '     o   '],
    ['  o      ', ' o       ', 'o        ', ' o       ', '  o      ', '   o     ', '    o    ', '     o   ', '      o  '],
    [' o       ', 'o        ', ' o       ', '  o      ', '   o     ', '    o    ', '     o   ', '      o  ', '       o '],
]

#for i, symbol in enumerate(cycle(loading_symbols)):
#    print(f"\033[u", end="")
#    for j in range(len(symbol)):
#        print(f"|{symbol[j]}|")
#    time.sleep(0.1)
#    if i >= 100:
#        break

def string_transformer(string: str, number_rep: str='subscript'):
    greek_letters: dict[str, str] = {
        'alpha': "\u03B1",
        'beta': "\u03B2",
        'gamma': "\u03B3",
        'delta': "\u03B4",
        'epsilon': "\u03B5",
        'zeta': "\u03B6",
        'eta': "\u03B7",
        'theta': "\u03B8",
        'iota': "\u03B9",
        'kappa': "\u03BA",
        'lambda': "\u03BB",
        'mu': "\u03BC",
        'nu': "\u03BD",
        'xi': "\u03BE",
        'omicron': "\u03BF",
        'pi': "\u03C0",
        'rho': "\u03C1",
        'sigma': "\u03C3",
        'tau': "\u03C4",
        'upsilon': "\u03C5",
        'phi': "\u03C6",
        'chi': "\u03C7",
        'psi': "\u03C8",
        'omega': "\u03C9",
    }
    superscripts: dict[str, str] = {
        '0': "\u2070",
        '1': "\u00B9",
        '2': "\u00B2",
        '3': "\u00B3",
        '4': "\u2074",
        '5': "\u2075",
        '6': "\u2076",
        '7': "\u2077",
        '8': "\u2078",
        '9': "\u2079",
    }
    subscripts: dict[str, str] = {
        '0': "\u2080",
        '1': "\u2081",
        '2': "\u2082",
        '3': "\u2083",
        '4': "\u2084",
        '5': "\u2085",
        '6': "\u2086",
        '7': "\u2087",
        '8': "\u2088",
        '9': "\u2089",
    }

    degrees: str = "\u00B1"
    plus_minus: str = "\u00B1"
    middle_dot: str = "\u00B7"
    left_angle_bracket: str = "\u27E8"
    right_angle_bracket: str = "\u27E9"
    nabla: str = "\u2207"
    product: str = "\u220F"
    summation: str = "\u2211"
    square_root: str = "\u221A"
    cube_root: str = "\u221B"
    proportional: str = "\u221D"
    infinity: str = "\u221E"
    partial_derivative: str = "\u2202"
    benzene: str = "\u232C"

    def string_modifier(key, value):
        return string.replace(key, value) if key in string else string
    
    for key, value in greek_letters.items():
        string = string_modifier(key, value)
    
    match number_rep.casefold():
        case 'superscript':
            for key, value in superscripts.items():
                string = string_modifier(key, value)
        
        case 'subscript':
            for key, value in subscripts.items():
                string = string_modifier(key, value)

        case _:
            pass

    #list(map(string_modifier, greek_letters.keys(), greek_letters.values()))
    #for i, j in greek_letters.items():
        #print(j.upper())

    print(string)
    #[print(chr(i)) for i in greek_letters.values()]

string_transformer("alpha1", number_rep='superscript')

