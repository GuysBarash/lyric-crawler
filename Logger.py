from datetime import datetime
from multiprocessing import Queue
from threading import Thread
from enum import Enum
import threading
import os


# Multiprocess\multithreaded logger
# By Guy Barash
# Documentation:

###########################################################
# initialization and setup
###########################################################
# import the classes LoggerLevel and Logger to use:
#
# from Logger import Logger
# from Logger import LoggerLevel
#
# To use initialize the logger:
# Create a logger ONLY in the main thread.
#
# if __name__ == '__main__':               #Thread handling cannot be done without asserting this
#      global logger                       #Must be global
#      logger = Logger()
#      fout_log = "output_report"           #Optional, add in order to create a copy of the console to file
#      logger.initThread(fout_log)          #the fout_log file is optional, here the thread is creates

###########################################################
# Send\Print generic text to logger:
###########################################################
# logger.log_print("This is an example text")
#
# the output will be printed:
# [ DEFAULT  ][      MAIN      ][General Flow][    0.0     ( 0 )] This is an example text
#
# [ DEFAULT  ]
#  is the level of the information (from LoggerLevel class) [      MAIN      ] is the role of the sender,
#  can be either [      MAIN      ],[    Thread-<ID>    ] or [PROCESS    <ID>]
#
# [General Flow]
# is which segment of the code is currently running, it's free speech with default value as
#  "General flow" , it can but should not exceed 10 characters.
#
# [    0.0     ( 0 )]
# The second number is the timer currently being use, by default it's timer 0 which started with the initialization
# of the logger, the first number is the time that has passed since the current timer has been restarted.

###########################################################
# finalizing
###########################################################
# only in the main thread, use:
# logger.log_close()

###########################################################
# Enable\Disable:
###########################################################
# logger.disable_logger()
# will prevent the logger from printing messages
# note that it is local, disabling in one thread will not affect the other.
# if the main is disabled during the split, the offsprings are disabled as well.
#
#
# logger.ensable_logger()
# will allow the logger to print messages after it was disabled.
# note that it is local, enabling in one thread will not affect the other.

###########################################################
# usage of timers
###########################################################
# by default only one timer exist , timer 0 , which starts its count upon init
# to create a new timer use \ change the default timer:
# logger.chooseTimer(<ID>)
#
# if the ID is new to the system, a new timer will be generated as started.
# if the ID is already in existence it will switch the main timer but will not set it to zero.
#
# To reset a timer use:
# resetTimer(<ID>)
# if the ID is new to the system, a new timer will be generated as started.
# if the ID is already in existence it will set it to zero.
#
# Using a specific timer for a specific message only:
# logger.log_print(msg , timer=<ID> )
# if the ID is new to the system, the time stamp will be from the current default
# if the ID is already in existence this message time will be according to timer <ID>

###########################################################
# Information levels
###########################################################
# Each message has its level , starting from default->debug->info->...->EXCEPTION (see LoggerLevel(Enum))
# sending a message with a level different from default can be done in one of two ways.
#
# Built in:
# instead of using logger.log_print , use:
# logger.log_debug("level is now Debug")
# logger.log_info("level is now Info")
# logger.log_warning("level is now Warning")
# logger.log_error("level is now Error")
# logger.log_critical("level is now Critical")
# logger.log_exception("level is now Expception :-( ")
#
# the regular print with a "level" flag:
# logger.log_print("The level here will be Debug",level=LoggerLevel.DEBUG)

###########################################################
# Modules
###########################################################
# different phases of the code have different purposes,
# To make the logger indicate the name of the phase\module use:
#
# logger.switcheModule("MODULE NAME")
#
# Output:
# [ DEFAULT  ][      MAIN      ][MODULE NAME][   0.016    ( 0 )] MODULE NAME
# The third box from now (until change) will mention the module name.
# The default value is "General Flow"
# To return to default wither use:
# logger.switcheModule("General Flow" )
# or:
# logger.switcheModuleToDefault()
#
# No need to declare the modules names before use.
# The string "__ALL__" is kept and should not be used!

###########################################################
# Filtering
###########################################################
# it is possible to force the logger to display messages only from a certain information level
# there is the general filter level and the per-module filter level.
# (see information levels in class LoggerLevel(Enum) to know values)
# To set the filtration level of a specific module (no need to declare the module before):
#
# logger.filter_module("<module name>", LoggerLevel.<level> )
#
# To set the general filtration level:
# logger.filter_general(LoggerLevel.<level>)
#
# each message sent to the logger will be compared with max(current_module_filter_level,general_filter_level)
# if that message level is LOWER than that - it will not be printed.



class LoggerLevel(Enum):
    DEFAULT = 0
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERR = 40
    CRITICAL = 50
    EXCEPTION = 60


class Msg:
    def __init__(self, str, timer=0, reset=False, chooseDefault=-1,
                 set_current_module=None,
                 level=LoggerLevel.DEFAULT,
                 module_to_filter=None,
                 module_to_filter_designated_level=LoggerLevel.DEFAULT):
        self.str = str
        self.timer = timer
        self.stamp = datetime.now()
        self.pid = os.getpid()
        self.tid = threading.current_thread().name
        self.reset = reset  # if reset is True then timer "timer" is reset to current time
        self.chooseDefault = chooseDefault  # does a new default timer required
        self.set_current_module = set_current_module
        self.level = level
        assert self.level in LoggerLevel, "ERROR: information level [{}] is not recognized".format(self.level)

        self.module_to_filter = module_to_filter
        self.module_to_filter_designated_level = module_to_filter_designated_level
        assert module_to_filter_designated_level in LoggerLevel, "ERROR: information level [{}] is not recognized".format(
            self.level)


class Logger:
    def __init__(self, q=None, level=None, module=None):
        self.log_queue = Queue() if (q is None) else q
        self.masterPid = os.getpid()
        self.masterTid = threading.current_thread().name

        self.enable = True  # If false msgs will not be created.

        self.timers = dict()
        self.timers[0] = datetime.now()
        self.defaultTimer = 0

        self.defaultModule = 'General Flow'
        self.currentModule = self.defaultModule
        self.modules = dict()
        self.modules[self.defaultModule] = LoggerLevel.DEFAULT
        self.modules["__ALL__"] = LoggerLevel.DEFAULT
        self.log_print("Logger initialized.", timer=0)

    # DISABLE logger:
    def disable_logger(self):
        self.enable = False

    def enable_logger(self):
        self.enable = True

    # Modules functions:
    def switcheModule(self, chosenModule):
        msg = Msg("Current module changed to {}".format(chosenModule), set_current_module=chosenModule)
        self.log_queue.put(msg)

    def switcheModuleToDefault(self):
        self.switcheModule(self.defaultModule)

    def filter_module(self, module, level):
        assert level in LoggerLevel, "ERROR: information level [{}] is not recognized".format(level)
        msg = Msg("Module {} filtering level is now {}".format(module, level.name),
                  module_to_filter=module,
                  module_to_filter_designated_level=level)
        self.log_queue.put(msg)

    def filter_general(self, level):
        assert level in LoggerLevel, "ERROR: information level [{}] is not recognized".format(level)
        msg = Msg("ALL modules filtering level is now {}".format(level.name),
                  module_to_filter="__ALL__",
                  module_to_filter_designated_level=level)
        self.log_queue.put(msg)

    # Timer functions
    def resetTimer(self, timer=0):
        msg = Msg("Clock {} Reset".format(timer), timer=timer, reset=True)
        self.log_queue.put(msg)

    def chooseTimer(self, timer=0):
        msg = Msg("TIMER {} SET A DEFAULT".format(timer), timer=timer, chooseDefault=timer)
        self.log_queue.put(msg)

    # Print functions
    def log_print(self, str="", level=LoggerLevel.DEFAULT, timer=None):
        if not self.enable:
            return

        t_timer = timer if timer != None else self.defaultTimer
        msg = Msg(str, timer=t_timer, level=level)
        self.log_queue.put(msg)

    def log_debug(self, str):
        self.log_print(str, level=LoggerLevel.DEBUG)

    def log_info(self, str):
        self.log_print(str, level=LoggerLevel.INFO)

    def log_warning(self, str):
        self.log_print(str, level=LoggerLevel.WARN)

    def log_error(self, str):
        self.log_print(str, level=LoggerLevel.ERR)

    def log_critical(self, str):
        self.log_print(str, level=LoggerLevel.CRITICAL)

    def log_exception(self, str):
        self.log_print(str, level=LoggerLevel.EXCEPTION)

    # Thread control
    def initThread(self, fout=None):
        if self.log_queue is None:
            self.log_queue = Queue()

        if Logger.log_t is None:
            Logger.initThread_static(fout, self)

    def log_close(self):
        if not (Logger.log_t is None):
            self.log_print("")
            self.log_print("Closing logger thread.")
            self.log_queue.put(None)
            Logger.log_t.join()
            Logger.log_t = None
            self.log_queue.close()
            self.log_queue = None
            if not (Logger.fout is None):
                Logger.fout.close()
                Logger.fout = None

    @staticmethod
    def initThread_static(fout, given_logger):
        global logger
        logger = given_logger
        if fout is None:
            Logger.fout = fout
        else:
            Logger.fout = open(fout, 'w')

        Logger.log_t = Thread(target=Logger.logger_thread, args=(logger.log_queue,))

        record = "[Info level][WORKER   ][ENVIRONMENT ][TIME     (TIMER )] Text msg."
        header = "--------------------------------------------------------------------------------"
        print record
        print header
        if (not (Logger.fout is None)) and (not Logger.fout.closed):
            Logger.fout.write("%s\n" % record)
            Logger.fout.write("%s\n" % header)

        Logger.log_t.start()

    @staticmethod
    def logger_thread(lq):
        global logger
        while True:
            pckg = lq.get()
            if pckg is None:
                break

            # Choose new default timer if needed
            if pckg.chooseDefault != -1:
                if not (pckg.chooseDefault in logger.timers.keys()):
                    logger.timers[pckg.chooseDefault] = pckg.stamp
                logger.defaultTimer = pckg.chooseDefault

            # Handle timer reset
            if pckg.reset:
                key = pckg.timer
                logger.timers[key] = pckg.stamp

            # Handle module switching:
            if not (pckg.set_current_module is None):
                logger.currentModule = pckg.set_current_module
                if not (logger.currentModule in logger.modules.keys()):
                    logger.modules[pckg.set_current_module] = LoggerLevel.DEFAULT
            level_stamp = "[{:^12}]".format(logger.currentModule)

            # Handle module filtering
            if not (pckg.module_to_filter is None):
                logger.modules[pckg.module_to_filter] = pckg.module_to_filter_designated_level

            # Actual filtering.
            filter_level = max(logger.modules[logger.currentModule].value, logger.modules["__ALL__"].value)
            if (filter_level > pckg.level.value) and (pckg.module_to_filter is None):
                continue

            # Determine main/process/thread
            role = "{:^9}".format("MAIN")
            if pckg.pid != logger.masterPid:
                role = "PID {:>5}".format(pckg.pid)
            elif pckg.tid != logger.masterTid:
                role = "TID {:>5}".format(pckg.tid.replace('Thread-', ''))

            # Determine time
            key = pckg.timer if pckg.timer in logger.timers.keys() else logger.defaultTimer
            timeDelta = (pckg.stamp - logger.timers[key]).total_seconds()
            timeStamp = "[ {0:^10} ({1:^3})]".format(timeDelta, key)

            # information level handling
            info_level_stamp = "[{:^10}]".format(pckg.level.name)

            record = "{4}[{1}]{3}{2} {0}".format(pckg.str, role, timeStamp, level_stamp, info_level_stamp)
            print record
            if (not (Logger.fout is None)) and (not Logger.fout.closed):
                Logger.fout.write("%s\n" % record)


Logger.log_t = None
Logger.fout = None
