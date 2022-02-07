import curses
from curses.textpad import Textbox, rectangle

"""
stdscr = curses.initscr()

# DO NOT PRINT KEYS ON SCREEN
curses.noecho()

#Able to react to keys without enter 
curses.cbreak()

# navigation
stdscr.keypad(True)

# TERMINATE
curses.nocbreak()
stdscr.keypad(False)
curses.echo()
curses.endwin()
"""
def main(stdscr):
    assert curses.has_colors() == True

    #Initiate color pairs
    curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_WHITE)
    curses.init_pair(2, curses.COLOR_RED,  curses.COLOR_BLACK)

    # Set cursor invisible
    curses.curs_set(0)
    
    # Clear screen
    stdscr.bkgd(' ', curses.color_pair(1))
    stdscr.clear()



    # This raises ZeroDivisionError when i == 10.
    for i in range(0, 10):
        v = i-10
        stdscr.addstr(i, 0, f'10 divided by {v} is {10/v}', curses.color_pair(1))
        c = stdscr.getch()

    stdscr.refresh()
    stdscr.getkey()

    begin_x = 20; begin_y = 7
    height = 5; width = 40
    win = curses.newwin(height, width, begin_y, begin_x)
    #rectangle(stdscr, 1, 0, height-1, width-1)
    win.border()
    win.addstr(1,1,'ERROR',curses.A_BOLD | curses.color_pair(2))

    win.refresh()
    win.getkey()
    win.refresh()
    win.getkey()
    win.refresh()
    win.getkey()


curses.wrapper(main)
