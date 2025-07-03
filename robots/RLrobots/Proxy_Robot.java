// Proxy robot is a proxy client that communicate with a Python gateway to send and receive state and action
// It is used to train the RL agent
package RLrobots;

import robocode.*;
import java.io.*;
import java.net.*;
import java.net.URI;
import javax.websocket.*;


public class Proxy_Robot extends AdvancedRobot implements WebSocketClient.MessageHandler {
    
    public WebSocketClient webSocket;

    private byte moveDirection = 1;
    private long lastFireTime = 0;
    public RobotState currentState;
    public RobotAction robotAction = new RobotAction(this);
    // Store pending action to execute in main robot thread
    private volatile int pendingAction = -1;
    private volatile boolean actionReady = false;
    
    public boolean isEnded = false;

    public void debug(String message){
        System.out.println("Tick: " + getTime() + " - " + message);
    }

    public void run() {
        try {
            // Connect to Python server via WebSocket
            webSocket  = new WebSocketClient();
            webSocket.setMessageHandler(this);
            setAdjustGunForRobotTurn(true);
	        setAdjustRadarForGunTurn(true);
            this.currentState = new RobotState(getTime());

            while (true) { 
                if (isEnded) {
                    debug("Robot ended" );
                    // webSocket.closeConnection();
                    break;
                }
                this.currentState.updateRobotState(getX(), getY(), getHeading(), getEnergy(), getGunHeading(), getGunHeat(), getVelocity(), getDistanceRemaining());
                webSocket.sendMessage(this.currentState.toJson());              

                while (!actionReady) { 
                    try {
                        Thread.sleep(1);
                    } catch (InterruptedException ex) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }

                if (actionReady) {
                    robotAction.executeAction(pendingAction);
                    actionReady = false; 
                }
                
                
                double enemyDistance = this.currentState.enemyDistance;
                
                this.currentState = new RobotState(getTime());

                // if (robotAction.hasFired(pendingAction)) {
                //     currentState.addReward(-2, "Fired gun");
                // }
                // currentState.addReward(-0.5, "Optimize for using less time");
                if (getGunHeat() != 0 && robotAction.hasFired(pendingAction)) {
                    currentState.addReward(-5, "Fired hot gun");
                }

                if (pendingAction == robotAction.AIM) {
                    if (enemyDistance == 0) {
                        currentState.addReward(-5, "Aim Wrong");
                    } else {
                        // double gunTurn = getHeading() - getGunHeading() + currentState.enemyBearing;
                        // if (gunTurn == 0) {
                        currentState.addReward(5, "Already Aimed");
                        // }
                    }
                }
                execute();
            }

        } catch (Exception e) {
            System.out.println("Error: " + e);
        } 

    }
    
    // Implement the MessageHandler interface
	@Override
	public void handleMessage(String message) {
        setPendingAction(Integer.parseInt(message));
    }

    public void onScannedRobot(ScannedRobotEvent e) {
        // debug("Scanned robot!");
        double enemyDistance = e.getDistance();
        double enemyHeading = e.getHeading();
        double enemyBearing = e.getBearing();
        double enemyVelocity = e.getVelocity();
        double absBearing = getHeading() + enemyBearing;
        double enemyX = getX() + enemyDistance * Math.sin(Math.toRadians(absBearing));
        double enemyY = getY() + enemyDistance * Math.cos(Math.toRadians(absBearing));
        this.currentState.updateEnemyState(enemyBearing, enemyDistance, enemyHeading, enemyX, enemyY, enemyVelocity);
    }
    
    // Modified handleMessage to set pending action instead of executing immediately
    private void setPendingAction(int action) {
        this.pendingAction = action;
        this.actionReady = true;
    }

	
	double normalizeBearing(double angle) {
		while (angle >  180) angle -= 360;
		while (angle < -180) angle += 360;
		return angle;
	}

    public void onBulletHit(BulletHitEvent e) {
        // Robot hit enemy
        this.currentState.addReward(10, "Hit enemy");
    }
    
    // public void onBulletHitBullet(BulletHitBulletEvent e) {
    //     // Robot's bullet hit enemy's bullet
    //     this.currentState.addReward(-2, "Bullet hit bullet");
    // }
    
    public void onHitRobot(HitRobotEvent e) {
        // Robot collided with another robot
        // debug("Hit robot");
        this.currentState.addReward(-2, "Hit robot");
    }
    
    public void onBulletMissed(BulletMissedEvent e) {
        // Robot's bullet missed the target
        // debug("Bullet missed");
        this.currentState.addReward(-2, "Bullet missed");
    }
    
    public void onHitWall(HitWallEvent e) {
        // Robot hit the wall
        // debug("Hit wall");
        this.currentState.addReward(-5, "Hit wall");
    }
    
    public void onHitByBullet(HitByBulletEvent e) {
        // Robot was hit by enemy
        // debug("Hit by enemy");
        this.currentState.addReward(-10, "Hit by enemy");
    }
    
    public void onWin(WinEvent e) {
        isEnded = true;
        webSocket.sendMessage("{\"isWin\": true, \"time\": " + getTime() + "}");
        // Robot won the battle
        debug("Robot won the battle");
    }
    
    public void onDeath(DeathEvent e) {
        isEnded = true;
        webSocket.sendMessage("{\"isWin\": false, \"time\": " + getTime() + "}");
        // Robot died
        debug("Robot died");

    }
    
    
    public void onRoundEnded(RoundEndedEvent e) {
        try{
            
        }catch(Exception ex){
            System.out.println("Error closing WebSocket connection: " + ex);
        }
        debug("Round ended");
    }
    
    public void onBattleEnded(BattleEndedEvent e) {
        debug("Battle ended");
    }
}