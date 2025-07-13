// Proxy robot is a proxy client that communicate with a Python gateway to send and receive state and action
// It is used to train the RL agent
package RLrobots;

import robocode.*;
import java.io.*;
import java.net.*;
import java.net.URI;
import javax.websocket.*;
import java.nio.ByteBuffer;
import java.awt.Color;
import java.awt.BasicStroke;
import java.awt.Graphics2D;

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
                    debug("Robot ended1" );
                    // webSocket.closeConnection();
                    break;
                }
                

                ByteBuffer imageBytes = getCurrentBattleView();
                webSocket.sendBinaryMessage(imageBytes);

                this.currentState.updateRobotState(getX(), getY(), getHeading(), getEnergy(), getGunHeading(), getGunHeat(), getVelocity(), getDistanceRemaining());
                webSocket.sendMessage(this.currentState.toJson());              

                
                double enemyDistance = this.currentState.enemyDistance;
                double enemyBearing = this.currentState.enemyBearing;
                boolean gunOnTarget = this.currentState.isGunOnTarget();
                boolean radarOnTarget = this.currentState.isRadarOnTarget();
                double gunTurn = getHeading() - getGunHeading() + enemyBearing;
                double gunHeat = getGunHeat();

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
                
                this.currentState = new RobotState(getTime());

                currentState.addReward(-1, "Taking time");

                // int firePower = robotAction.isFired(pendingAction);
                // if  (firePower > 0) {
                //     System.out.println("Fire " + (gunHeat) + " " + firePower);    
                //     if (gunHeat!=0) {
                //         currentState.addReward(-5, "Fired hot gun");
                //     } else {
                //          currentState.addReward(-firePower + 3, "Fire");
                //     }

                //     if (enemyDistance != 0 && Math.abs(normalizeBearing(gunTurn)) < 1) {
                //         System.out.println("Fire Right");
                //         currentState.addReward(10, "Fire Right");
                //     } else {
                //         System.out.println("Fire Wrong");
                //     }
                // }
            
                // if (pendingAction == robotAction.AIM) {
                //     if (!radarOnTarget) {
                //         System.out.println("Aim Wrong");
                //         currentState.addReward(-2, "Aim Wrong");
                //     } else {
                //         if (gunOnTarget) {
                //             System.out.println("Already Aimed");
                //             currentState.addReward(-5, "Already Aimed");
                //         }
                //         currentState.addReward(3, "Aimed Right");
                //     }
                // }

                execute();
            }

        } catch (Exception e) {
            System.out.println("Error1: " + e);
        } 

    }
    
    // Implement the MessageHandler interface
	@Override
	public void handleMessage(String message) {
        setPendingAction(Integer.parseInt(message));
    }

    public void onPaint(Graphics2D g) {
		g.setColor(Color.red);
		g.drawOval((int) (getX() - 25), (int) (getY() - 25), 50, 50);
		g.setColor(new Color(0, 0xFF, 0, 30));
		g.drawOval((int) (getX() - 25), (int) (getY() - 25), 50, 50);
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
        double bulletPower = e.getBullet().getPower();
        int reward = (int)(bulletPower * 3); // Scale reward based on bullet power
        this.currentState.addReward(reward, "Hit enemy with power " + bulletPower);
    }
    
    // public void onBulletHitBullet(BulletHitBulletEvent e) {
    //     // Robot's bullet hit enemy's bullet
    //     this.currentState.addReward(-2, "Bullet hit bullet");
    // }
    
    public void onHitRobot(HitRobotEvent e) {
        // Robot collided with another robot
        // debug("Hit robot");
        this.currentState.addReward(-1, "Hit robot");
    }
    
    public void onBulletMissed(BulletMissedEvent e) {
        // Robot's bullet missed the target
        // debug("Bullet missed");
        double bulletPower = e.getBullet().getPower();
        int reward = (int)(bulletPower * 2); // Scale reward based on bullet power
        this.currentState.addReward(-reward, "Bullet missed");
    }
    
    public void onHitWall(HitWallEvent e) {
        // Robot hit the wall
        // debug("Hit wall");
        this.currentState.addReward(-1, "Hit wall");
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