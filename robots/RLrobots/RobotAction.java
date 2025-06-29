package RLrobots;

import robocode.AdvancedRobot;
import robocode.Robot;
import java.util.List;
import java.util.ArrayList;

/**
 * RobotAction class handles all robot movement and combat actions
 * Separates action logic from the main robot class for better organization
 */
public class RobotAction {
    


    public static final int DO_NOTHING = 0;
    public static final int RUN_AWAY_RIGHT = 1;
    public static final int RUN_AWAY_LEFT = 2;
    public static final int RUN_AWAY_BACK = 3;
    public static final int RUN_AHEAD = 4;
    public static final int FIRE_1 = 5;
    public static final int FIRE_2 = 6;
    public static final int FIRE_3 = 7;
    public static final int AIM = 8;

    private Proxy_Robot robot;
    
    public RobotAction(Proxy_Robot robot) {
        this.robot = robot;
    }

    public boolean hasFired(int action){
        if (action == FIRE_1 || action == FIRE_2 || action == FIRE_3) {
            return true;
        }
        return false;
    }

    public void debug(String message) {
        //System.out.println(message);
    }

    public void executeAction(int action) {
        switch (action) {
            case DO_NOTHING:
                debug("Executing action: " + action + " (doNothing)");
                doNothing();
                break;
            case RUN_AWAY_RIGHT:
                debug("Executing action: " + action + " (runAwayRight)");
                runAwayRight();
                break;  
            case RUN_AWAY_LEFT:
                debug("Executing action: " + action + " (runAwayLeft)");
                runAwayLeft();
                break;
            case RUN_AWAY_BACK:
                debug("Executing action: " + action + " (runAwayBack)");
                runAwayBack();
                break;
            case RUN_AHEAD:
                debug("Executing action: " + action + " (runAhead)");
                runAhead();
                break;
            case FIRE_1:
                debug("Executing action: " + action + " (fire1)");
                fire1();
                break;
            case FIRE_2:
                debug("Executing action: " + action + " (fire2)");
                fire2();
                break;
            case FIRE_3:
                debug("Executing action: " + action + " (fire3)");
                fire3();
                break;
            case AIM:
                debug("Executing action: " + action + " (aim)");
                aim();
                break;
            default:
                debug("Unknown action: " + action);
                break;
        }
    }
    

    public void doNothing() {}

    public void runAwayRight() {
        robot.setTurnRight(robot.currentState.heading + 45);   
    }   

    public void runAwayLeft() {
        robot.setTurnRight(robot.currentState.heading - 45);   
    }

    public void runAwayBack() {
        robot.setAhead(-50);
    }

    public void runAhead() {
        robot.setAhead(50);
    }

    public void fire1() {
        robot.fire(1);
    }

    public void fire2() {
        robot.fire(2);
    }

    public void fire3() {
        robot.fire(3);
    }

    public void aim() {
        double gunTurn = robot.getHeading() - robot.getGunHeading() + robot.currentState.enemyBearing;
        robot.setTurnGunRight(normalizeBearing(gunTurn));
    }
    
    public void aim_90() {
        double gunTurn = robot.getHeading() - robot.getGunHeading() + robot.currentState.enemyBearing;
        robot.setTurnGunRight(normalizeBearing(gunTurn)+ 90);
    }

        
    double normalizeBearing(double angle) {
        while (angle >  180) angle -= 360;
        while (angle < -180) angle += 360;
        return angle;
    }

    
} 
