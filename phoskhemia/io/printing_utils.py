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
bouncing_loader(width='auto', symbol=inchworm)

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
