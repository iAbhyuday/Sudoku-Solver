from puzzle import extract_digits,find_puzzle
import imutils
import cv2
import numpy as np
from sudoku import Sudoku
import torch
import sys
# model
model = torch.load("sudoku.pth")

def Solve(image):
    
    image = imutils.resize(image,width=600)
    
    puzzleImage,warped_image = find_puzzle(image,debug=False)

    board = np.zeros((9,9),dtype='int')

    stepX = warped_image.shape[1]//9
    stepY = warped_image.shape[0]//9

    cellLocs = []
    
    for y in range(0,9):
        row=[]

        for x in range(0,9):

            startX = x*stepX
            startY = y*stepY

            endX = (x + 1)*stepX
            endY = (y + 1)*stepY

            # row.append((startX,startY,endX,endY))

            cell = warped_image[startY:endY,startX:endX]

            digit = extract_digits(cell,debug=False)




            if digit is not None:

                roi = cv2.resize(digit,(28,28))
                roi = torch.from_numpy(roi).float()
                roi = roi.reshape((1,1,28,28))

                out = model(roi)
                _,pred = torch.max(out,1)


                board[y,x] = pred.data+1
                row.append((-1,-1,-1,-1))
            else:
                row.append((startX,startY,endX,endY))
        cellLocs.append(row)



    print("[INFO] OCR'd Sudoku board:")
    puzzle = Sudoku(3, 3, board=board.tolist())
    puzzle.show()

    solution = puzzle.solve()

    for (cellRow,boardRow) in zip(cellLocs,solution.board):
        for (box,digit) in zip (cellRow,boardRow):

            if(box==(-1,-1,-1,-1)):
                pass
            else:
                startX,startY,endX,endY = box
                
                textX = int((endX-startX)*0.33)
                textY = int((endY-startY)* -0.2)

                textX+=startX
                textY+=endY

                cv2.putText(puzzleImage,str(digit),(textX,textY),cv2.FONT_HERSHEY_SIMPLEX
                ,0.9,(0,0,255),2)
        cv2.imshow("Result",puzzleImage)
    cv2.waitKey(0)

 if __name__=='__main__':
    image = sys.argv[1]
    im = cv2.imread(image)

    if im is None:
        print("[Error] Image location is incorrect!\n[INFO] Use only JPG/PNG/JPEG images\n")
    else:
        Solve(im)
