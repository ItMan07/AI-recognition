# ======================================= ИМПОРТ БИБЛИОТЕК ==============================

import re

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QIODevice
from PyQt5.QtSerialPort import QSerialPortInfo, QSerialPort
from PyQt5.QtWidgets import QMessageBox

# import voiceAssistant as va

import recognition.hands_recognition as hr
import recognition.haar_cascade_recognition as fr

# ======================================= НАСТРОЙКИ =====================================


app = QtWidgets.QApplication([])
ui = uic.loadUi("ai program.ui")
ui.setWindowTitle("AI UI")

flagBtnLed = False

serial = QSerialPort()
serial.setBaudRate(9600)
portList = []

strON = "🟢 Включено"
strOFF = "🔴 Выключено"
stop = False


# ======================================= ФУНКЦИИ =======================================


def OnRead():
    # if not serial.canReadLine(): return  # выходим если нечего читать
    rx = serial.readLine()
    rx = str(rx, 'utf-8').strip()
    data = re.split("[,;]", rx)

    # data = rx
    print(data)

    if data[0] == '100':  # ПОДКЛЮЧЕНИЕ
        if data[1] == '1':
            openSuccess()
            print("Устройство подключено")

    if data[0] == '0':
        if data[1] == '0':
            ui.labelLed.setText(strOFF)
        elif data[1] == '1':
            ui.labelLed.setText(strON)


def onOpen():
    serial.setPortName(ui.comPortsCB.currentText())
    serial.open(QIODevice.ReadWrite)


def OnClose():
    serial.close()
    # ui.connectBtn.setText(strOFF)


def serialSend(data):
    txs = ""
    for val in data:
        txs += str(val) + ','
    txs = txs[:-1]
    txs += ';'
    serial.write(txs.encode())


def reloadComPortsCB():
    ui.comPortsCB.clear()
    ports = QSerialPortInfo().availablePorts()

    global portList
    portList = []
    for port in ports:
        portList.append(port.portName())
        # portList.append(port.systemLocation())
        portList.append(port.description())
    #

    ui.comPortsCB.addItems(portList)


def openSuccess():
    info = QMessageBox()
    info.setWindowTitle("Информация")
    info.setText("COM-порт был успешно открыт! \n")
    info.setInformativeText("Arduino устройство готово к работе")
    info.setIcon(QMessageBox.Information)
    info.setStandardButtons(QMessageBox.Ok)
    ui.connectBtn.setText(strON)
    info.exec_()


def enable_hands_recognition():
    hr.start_hr()


def enable_face_recognition():
    fr.start_fr()


# ======================================= ИНТЕРФЕЙС =====================================


def load_UI():
    # ui.comPortsCB.addItems(portList)
    reloadComPortsCB()
    ui.reloadBtn.clicked.connect(reloadComPortsCB)
    serial.readyRead.connect(OnRead)
    ui.openBtn.clicked.connect(onOpen)
    ui.closeBtn.clicked.connect(OnClose)

    ui.handsBtn.clicked.connect(enable_hands_recognition)
    ui.faceBtn.clicked.connect(enable_face_recognition)

    ui.show()
    app.exec()


if __name__ == "__main__":
    load_UI()
