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
    private volatile double pendingValue = 0.0;
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
            webSocket.sendMessage("{\"gameStart\": false, \"time\": " + getTime() + "}");
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
                    debug("Executing action: " + pendingAction + " with value: " + pendingValue);
                    robotAction.executeAction(pendingAction, pendingValue);
                    actionReady = false; 
                }
                
                
                double enemyDistance = this.currentState.enemyDistance;
                double enemyBearing = this.currentState.enemyBearing;
                double gunTurn = this.currentState.heading - this.currentState.gunHeading + enemyBearing;
                double gunHeat = this.currentState.gunHeat;

                this.currentState = new RobotState(getTime());

                // if (robotAction.hasFired(pendingAction)) {
                //     currentState.addReward(-2, "Fired gun");
                // }
                // currentState.addReward(-0.5, "Optimize for using less time");
                currentState.addReward(-3, "Do nothing");

                if (robotAction.isFired(pendingAction) != 0) {
                    if (gunHeat != 0) {
                        currentState.addReward(-5, "Fired hot gun");
                    } else {
                        if (gunTurn == 0) { // 
                            currentState.addReward(5, "Fired cold gun");
                        } else {
                            currentState.addReward(3, "Fired cold gun");
                        }
                    }
                }

                if (pendingAction == robotAction.AIM) {
                    if (enemyDistance == 0) {
                        currentState.addReward(-5, "Aim Wrong");
                    } else {
                        if (gunTurn != 0) { 
                            currentState.addReward(5, "Aimed Right");
                        } 
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
        try {
            // Parse JSON message from Python PPO agent
            // Expected format: {"action": 3, "action_prob": 0.25, "value": 0.5}
            if (message.trim().startsWith("{")) {
                // JSON format - extract action and value
                int actionStart = message.indexOf("\"action\":") + 9;
                int actionEnd = message.indexOf(",", actionStart);
                if (actionEnd == -1) {
                    actionEnd = message.indexOf("}", actionStart);
                }
                String actionStr = message.substring(actionStart, actionEnd).trim();
                
                // Extract value if present
                double value = 0.0;
                int valueStart = message.indexOf("\"value\":");
                if (valueStart != -1) {
                    valueStart += 8;
                    int valueEnd = message.indexOf(",", valueStart);
                    if (valueEnd == -1) {
                        valueEnd = message.indexOf("}", valueStart);
                    }
                    String valueStr = message.substring(valueStart, valueEnd).trim();
                    try {
                        value = Double.parseDouble(valueStr);
                    } catch (NumberFormatException e) {
                        System.out.println("Error parsing value: " + valueStr);
                        value = 0.0;
                    }
                }
                System.out.println("Action: " + actionStr + " Value: " + value);
                setPendingActionAndValue(Integer.parseInt(actionStr), value);
            } else {
                // Legacy format - direct integer
                setPendingActionAndValue(Integer.parseInt(message), 0.0);
            }
        } catch (Exception e) {
            System.out.println("Error parsing action message: " + message + " - " + e.getMessage());
            // Default to no action (0) if parsing fails
            setPendingActionAndValue(0, 0.0);
        }
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
    
    // Modified handleMessage to set pending action and value instead of executing immediately
    private void setPendingActionAndValue(int action, double value) {
        this.pendingAction = action;
        this.pendingValue = value;
        this.actionReady = true;
    }
    
    // Legacy method for backward compatibility
    private void setPendingAction(int action) {
        setPendingActionAndValue(action, 0.0);
    }

	
	double normalizeBearing(double angle) {
		while (angle >  180) angle -= 360;
		while (angle < -180) angle += 360;
		return angle;
	}

    public void onBulletHit(BulletHitEvent e) {
        // Robot hit enemy
        double bulletPower = e.getBullet().getPower();
        int reward = (int)(bulletPower * 10); // Scale reward based on bullet power
        this.currentState.addReward(reward, "Hit enemy with power " + bulletPower);
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
        // this.currentState.addReward(-2, "Bullet missed");
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