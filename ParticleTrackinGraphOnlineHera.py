import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

#Laurent Berthe @labPIMM - 07/2025 - Laurent.berthe (at) Cnrs.fr
#Select particle and extract, graph and print on line velocity from .avi video. 
#Inputs : 
#    FrameSpaceScale: space scale in Unit/px
#    DeltaTimeFrame : delta time between frame in time uit
#    VideoName: Name of Video file 

#On Camera window keyboard  
#'q' : quit and final graph
# p increase threshold detection on gray scale image 
# m decrease thershold detection on gray sacale image
# -> Frame advance
# <- Frame back
# Range : scale on image in um

VideoName="VideoTest.avi"
DeltaTimeFrame=100e-9 #s
FrameSpaceScale=8.36 #(um/px)
Range=100 #(um) 

#Function for particle selection
def click(event, x, y, flags, param):
    global CurrentRoi_x, CurrentRoi_y, CurrentRoi_w, CurrentRoi_h, PreviousCenterTrackingX, PreviousCenterTrackingY,SelectionDone, VideoFrame, VideoFrame
    FrameToPick=VideoFrame
    if event==cv2.EVENT_LBUTTONDBLCLK:
        #Select windown on picture, using mouse left bouton, space to quit, C to cancel
        CurrentRoi_x, CurrentRoi_y, CurrentRoi_w, CurrentRoi_h=cv2.selectROI('Particle Selection', FrameToPick, False, False)
        #udpate previous tracking for velocity calculation
        PreviousCenterTrackingX = CurrentRoi_x + CurrentRoi_w // 2
        PreviousCenterTrackingY = CurrentRoi_y + CurrentRoi_h // 2
        #confirm that there is an active selection
        SelectionDone=1

def GraphPreperation ():
    global fig,axs,line_pos_x,line_pos_y,regression_lineX,regression_lineY,line_vel_x,line_vel_y,line_vel_pos_x,line_vel_pos_y
    plt.ion()  # Mode interactive
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))

    # Prepare graph
    line_pos_x, = axs[0].plot([], [], 'r.-', label='X position')
    line_pos_y, = axs[0].plot([], [], 'b.-', label='Y position')
    regression_lineX, = axs[0].plot([], [], 'k--', label='Fit X(t)')
    regression_lineY, = axs[0].plot([], [], 'k--', label='Fit Y(t)')
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position (um)")
    axs[0].set_title("Position vs Time")
    axs[0].legend()
    axs[0].grid(True)

    line_vel_x, = axs[1].plot([], [], 'r.-', label='Velocity X')
    line_vel_y, = axs[1].plot([], [], 'b.-', label='Velocity Y')
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_title("Velocity vs Time")
    axs[1].legend()
    axs[1].grid(True)

    line_vel_pos_x, = axs[2].plot([], [], 'r.-', label='X')
    line_vel_pos_y, = axs[2].plot([], [], 'b.-', label='Y')
    axs[2].set_xlabel("Position (um)")
    axs[2].set_ylabel("Velocities (m/s)")
    axs[2].set_title("Velocities vs Position")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()

#scale on Image
longueur_pixels = int(Range/(FrameSpaceScale))
x_start, y_start = 10, 240
x_end, y_end = x_start + longueur_pixels, y_start

#Unit Si
FrameSpaceScale=FrameSpaceScale*1e-6

#Inititate datas
ParticuleXPositions = []
ParticuleYPositions = []
VelocityXPositions = []
VelocityYPositions = [] 
TimeFrame=[]
CurrentCenterTrackingX=150
CurrentCenterTrackingY=150
VelocityX=0
VelocityY=0
mean_Vx=0
mean_Vy=0

SelectionDone=0
#nbr_classes=180
VideoThreshold=50 # white/black treshold detection. 
CurrentFrame = 0
#Search criteria
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1.0)

#Window naming
cv2.namedWindow('Camera Raw')
cv2.namedWindow('Camera Gray')
cv2.namedWindow("BinaryTracking")

# Data extraction from image. 
video=cv2.VideoCapture(VideoName)
TotalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
WidthFrame = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
HeightFrame = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
FpsFrame = video.get(cv2.CAP_PROP_FPS)
DurationFrame = TotalFrames / FpsFrame if FpsFrame else 0

print ('Images properties')
print(f"Total frames : {TotalFrames}")
print(f"Dimensions : {WidthFrame}x{HeightFrame}")
print(f"FPS : {FpsFrame}")
print(f"Durée (s) : {DurationFrame:.2f}")

#Graph Preparation
GraphPreperation()

#Window particle selection
cv2.setMouseCallback('Camera Raw', click)

#Loop to extact, follow graph particule position and velocity
while True:
    #Goto Currentframe
    video.set(cv2.CAP_PROP_POS_FRAMES, CurrentFrame)
    ret, VideoFrame = video.read()
    #Clean Info Window at each frame 
    InfoWindow = np.ones((600, 700, 3), dtype=np.uint8) * 255
    if not ret:
        break
    #Transform to gray scale image
    VideoGray= cv2.cvtColor(VideoFrame, cv2.COLOR_BGR2GRAY) 
    #if active selection 
    if SelectionDone:
        #Goto to blanck/white binary image at VideoThreshold
        _, mask =cv2.threshold(VideoGray, VideoThreshold, 255, cv2.THRESH_BINARY_INV)
        
        #Emage cleaning
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=3)
        #Draw Video Binary mask
        cv2.imshow('BinaryTracking', mask)
        
        #Particle searching
        _, rect = cv2.meanShift(mask, (CurrentRoi_x, CurrentRoi_y, CurrentRoi_w, CurrentRoi_h), term_criteria)
        CurrentRoi_x, CurrentRoi_y, CurrentRoi_w, CurrentRoi_h = rect
        
        #Graph particle windows
        cv2.rectangle(VideoFrame, (CurrentRoi_x, CurrentRoi_y), (CurrentRoi_x + CurrentRoi_w, CurrentRoi_y + CurrentRoi_h), (0, 0, 255), 2)
        cv2.rectangle(VideoGray, (CurrentRoi_x, CurrentRoi_y), (CurrentRoi_x + CurrentRoi_w, CurrentRoi_y + CurrentRoi_h), (0, 0, 255), 2)
        cv2.rectangle(mask, (CurrentRoi_x, CurrentRoi_y), (CurrentRoi_x + CurrentRoi_w, CurrentRoi_y + CurrentRoi_h), (255, 255, 255), 2)
        
        #Calculate Central Position
        CurrentCenterTrackingX = CurrentRoi_x + CurrentRoi_w // 2
        CurrentCenterTrackingY = CurrentRoi_y + CurrentRoi_h // 2

        #Velocity calculation
        VelocityX=abs((PreviousCenterTrackingX-CurrentCenterTrackingX)*FrameSpaceScale/DeltaTimeFrame)
        VelocityY=abs((PreviousCenterTrackingY-CurrentCenterTrackingY)*FrameSpaceScale/DeltaTimeFrame)
        
        #PreviousParticles defupdate
        PreviousCenterTrackingX=CurrentCenterTrackingX
        PreviousCenterTrackingY=CurrentCenterTrackingY
        
        print ('velocityX :', VelocityX)
        print ('velocityY :', VelocityY)
        
        #SaveForGraph*******************
        ParticuleXPositions.append(CurrentCenterTrackingX*FrameSpaceScale)
        ParticuleYPositions.append(CurrentCenterTrackingY*FrameSpaceScale)
        VelocityXPositions.append(VelocityX)
        VelocityYPositions.append(VelocityY) 
        TimeFrame.append(CurrentFrame*DeltaTimeFrame)
        
        #Mean velocities
        Vx = np.array(VelocityXPositions)
        Vy = np.array(VelocityYPositions)

        # Mean velocities
        mean_Vx = np.mean(Vx)
        mean_Vy = np.mean(Vy)

        # Resultante
        mean_V = np.mean(np.sqrt(Vx**2 + Vy**2))
        print(f"Vitesse moyenne en X : {mean_Vx:.3e} m/s")
        print(f"Vitesse moyenne en Y : {mean_Vy:.3e} m/s")
        print(f"Vitesse moyenne résultante : {mean_V:.3e} m/s")
        
        #graph on line position, velocities
        
        if len(ParticuleXPositions) >= 2:
            slope, intercept = np.polyfit(TimeFrame, ParticuleXPositions, 1)
            TimeFrame_fit = np.linspace(min(TimeFrame), max(TimeFrame), 50)
            ParticuleXPositions_fit = slope * TimeFrame_fit + intercept
            regression_lineX.set_data(TimeFrame_fit, ParticuleXPositions_fit)
            regression_lineX.set_label(f"Fit X: X = {slope:.2f}Time + {intercept:.2f}")
            line_pos_x.set_data(TimeFrame, ParticuleXPositions)
            line_pos_y.set_data(TimeFrame, ParticuleYPositions)
            axs[0].legend()
        
        if len(ParticuleYPositions) >= 2:
            slopeY, interceptY = np.polyfit(TimeFrame, ParticuleYPositions, 1)
            TimeFrame_fit = np.linspace(min(TimeFrame), max(TimeFrame), 50)
            ParticuleYPositions_fit = slopeY * TimeFrame_fit + interceptY
            regression_lineY.set_data(TimeFrame_fit, ParticuleYPositions_fit)
            regression_lineY.set_label(f"Fit Y: Y = {slopeY:.2f}Time + {interceptY:.2f}")
            axs[0].legend()
        
        line_vel_x.set_data(TimeFrame, VelocityXPositions)
        line_vel_y.set_data(TimeFrame, VelocityYPositions)

        line_vel_pos_x.set_data(ParticuleXPositions, VelocityXPositions)
        line_vel_pos_y.set_data(ParticuleXPositions, VelocityYPositions)
                        
        # adjust axis 
        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        # refresh figures 
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001) 
               
                 
        
    #Print value on WindowInfo
    cv2.putText(InfoWindow, f"ParticlePosition: (H : { CurrentCenterTrackingX} px, V: {CurrentCenterTrackingY} px)", (10, 70),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 2)
        
    cv2.putText(InfoWindow, f" VelocityX : {VelocityX} (Mean : {mean_Vx}) m/s ", (10, 95),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 2)
    cv2.putText(InfoWindow, f" VelocityY : {VelocityY} (Mean : {mean_Vy}) m/s " , (10, 120),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 2)
        
            
    cv2.putText(InfoWindow, f"Frame [Cmd :<-|->]: {CurrentFrame}/{TotalFrames}", (10, 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 2)
    
    cv2.putText(InfoWindow, f"VideoThreshold [Cmd : p|m] : {VideoThreshold}", (10, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 2)
    
    cv2.putText(InfoWindow, f"FrameSpaceScale : {FrameSpaceScale} um/px", (10, 145),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 165, 255), 2)
    cv2.putText(InfoWindow, f"DeltaTimeFrame: {DeltaTimeFrame} s", (10, 170),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 165, 255), 2)
    
    cv2.putText(InfoWindow, "Video File : "+ VideoName, (10, 195),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 165, 255), 2)
    
    cv2.putText(InfoWindow, "Particle selection : Dbl click on Raw and space to back to CamRaw" , (10, 220),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 2)
    
    cv2.putText(InfoWindow, "q to quit" , (10, 245),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 2)
    
    #Scale on video Raw
    cv2.line(VideoFrame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 1)  # vert, épaisseur 3
    cv2.putText(VideoFrame, f'{Range} um', (x_start, y_start - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    #Update windows
    cv2.imshow("InfoTracking", InfoWindow)
    cv2.imshow("Camera Gray", VideoGray)
    cv2.imshow('Camera Raw', VideoFrame)
    
    #Key active for operation on windows
    key = cv2.waitKey(0)
    

    if key == ord('q'):
     
        break
    elif key == ord('p'): #Increase Videos Binary transition threshold
        VideoThreshold = min(250, VideoThreshold + 1)
        print(f" VideoThreshold +: {VideoThreshold}")
        continue
    elif key == ord('m'): #decrease Videos Binary transition threshold
        VideoThreshold = max(1, VideoThreshold - 1)
        print(f" VideoThreshold -: {VideoThreshold}")
    elif key == 81:  # ← #Time Frame decrease
        CurrentFrame = max(0, CurrentFrame - 1)
        print(f"CurrentFrame {CurrentFrame}")
        continue
    elif key == 83:  # → #Time Frame increase
        CurrentFrame = min(TotalFrames - 1, CurrentFrame+1)
        print(f"CurrentFrame {CurrentFrame}")
        continue

video.release()
cv2.destroyAllWindows()

