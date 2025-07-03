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
    


    public static final int DO_NOTHING = 1000;
    public static final int RUN_AWAY_RIGHT = 0;
    public static final int RUN_AWAY_LEFT = 1;
    public static final int RUN_AWAY_BACK = 2;
    public static final int RUN_AHEAD = 3;
    public static final int FIRE_1 = 4;
    public static final int FIRE_2 = 66;
    public static final int FIRE_3 = 77;
    public static final int AIM = 5;
    public static final int TURN_LEFT = 99;
    public static final int TURN_RIGHT = 100;
    
    public int radar_direction = 1;

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
        System.out.println(message);
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
            // case CHANGE_RADAR_DIRECTION:
            //     radar_direction = radar_direction * -1;
            //     break;
            case TURN_LEFT:
                debug("Executing action: " + action + " (turnLeft)");
                turnLeft();
                break;
            case TURN_RIGHT:
                debug("Executing action: " + action + " (turnRight)");
                turnRight();
                break;
            default:
                System.out.println("Unknown action: " + action);
                System.exit(0);
                break;
        }
        robot.setTurnRadarRight(radar_direction * 90);
    }
    

    public void doNothing() {}

    public void runAwayRight() {
        robot.setTurnRight(45);  
    }   

    public void runAwayLeft() {
        robot.setTurnLeft(45);   
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
    
    double normalizeBearing(double angle) {
        while (angle >  180) angle -= 360;
        while (angle < -180) angle += 360;
        return angle;
    }

    public void turnLeft() {
        robot.setTurnRight(robot.currentState.heading - 90);   
    }

    public void turnRight() {
        robot.setTurnRight(robot.currentState.heading + 90); 
    }
} 
