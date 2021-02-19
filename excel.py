import xlwings as xw

class Excel(object):
    def __init__(self):
        # Initial the parameter of excel
        print("Initial the parameter of excel.")
        self.app = xw.App(visible=True, add_book=False)
        self.wb = self.app.books.add()
        self.sheet = self.wb.sheets[0]

    def write_excel(self, start, content, vertical=False):
        print('write the content.\n')
        self.sheet.range(start).options(transpose=vertical).value = content

    def save_excel(self, file_name):
        print('save the excel file.')
        self.wb.save(file_name)

    def close_excel(self):
        self.wb.close()
        self.app.quit()

    def clear_excel(self):
        self.sheet.clear()

    def write_loss_and_iou(self, record, loss, val_loss, iou, avr_iou):
        self.write_excel('a1', record)
        self.write_excel('a2', 'loss')
        self.write_excel('b2', 'val_loss')
        self.write_excel('a3', loss, vertical=True)
        self.write_excel('b3', val_loss, vertical=True)
        self.write_excel('c2', 'iou')
        self.write_excel('d2', 'avr_iou')
        self.write_excel('c3', iou, vertical=True)
        self.write_excel('d3', avr_iou, vertical=True)
