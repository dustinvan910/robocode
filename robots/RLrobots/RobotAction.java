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
    
    // action for moving
    public static final int FACING_ENEMY = 0;
    public static final int TURN_90_ENEMY = 1;
    public static final int RUN_CLOCKWISE = 2;
    public static final int RUN_COUNTER_CLOCKWISE = 3;
    public static final int RUN_AWAY_BACK = 4;
    public static final int RUN_AHEAD = 5;

    // action for aiming and firing
    public static final int FIRE_AUTO = 6;
    public static final int AIM_GUN = 7;

    public static final int FACING_ENEMY_FIRE = 77;
    public static final int TURN_90_ENEMY_FIRE = 78;
    public static final int RUN_CLOCKWISE_FIRE = 79;
    public static final int RUN_COUNTER_CLOCKWISE_FIRE = 80;  
    public static final int RUN_AWAY_BACK_FIRE = 81;
    public static final int RUN_AHEAD_FIRE = 82;

    public static final int FIRE_1 = 83;
    public static final int FIRE_2 = 84;
    public static final int FIRE_3 = 85;
    // public static final int TURN_LEFT = 88;
    // public static final int TURN_RIGHT = 100;
    
    public int move_direction = 1;

    private Proxy_Robot robot;
    
    public RobotAction(Proxy_Robot robot) {
        this.robot = robot;
    }

    public int isFired(int action){
        if (action == FIRE_1) {
            return 1;
        } else if (action == FIRE_2) {
            return 2;
        } else if (action == FIRE_3) {
            return 3;
        }
        if (action == FIRE_AUTO) {
            return 1;
        }
        return 0;
    }

    public void debug(String message) {
        System.out.println(message);
    }

    public void executeAction(int action) {
        switch (action) {
            case FACING_ENEMY_FIRE:
                debug("Executing action: " + action + " (facingEnemyFire)");
                facingEnemy();
                fireAuto();
                break;
            case FACING_ENEMY:
                debug("Executing action: " + action + " (facingEnemy)");
                facingEnemy();
                break;
            case TURN_90_ENEMY:
                debug("Executing action: " + action + " (turn90Enemy)");
                ninetyDegrees();
                break;
            case TURN_90_ENEMY_FIRE:
                debug("Executing action: " + action + " (turn90EnemyFire)");
                ninetyDegrees();
                fireAuto();
                break;
            case RUN_CLOCKWISE:
                debug("Executing action: " + action + " (runClockwise)");
                runClockwise();
                break;  
            case RUN_COUNTER_CLOCKWISE:
                debug("Executing action: " + action + " (runCounterClockwise)");
                runCounterClockwise();
                break;
            case RUN_CLOCKWISE_FIRE:
                debug("Executing action: " + action + " (runClockwiseFire)");
                runClockwise();
                fireAuto();
                break;  
            case RUN_COUNTER_CLOCKWISE_FIRE:
                debug("Executing action: " + action + " (runCounterClockwiseFire)");
                runCounterClockwise();
                fireAuto();
                break;
            case RUN_AWAY_BACK_FIRE:
                debug("Executing action: " + action + " (runAwayBackFire)");
                runAwayBack();
                fireAuto();
                break;
            case RUN_AHEAD_FIRE:
                debug("Executing action: " + action + " (runAheadFire)");   
                runAhead();
                fireAuto();
                break;
            case RUN_AWAY_BACK:
                debug("Executing action: " + action + " (runAwayBackFire)");
                runAwayBack();
                break;
            case RUN_AHEAD:
                debug("Executing action: " + action + " (runAhead)");
                runAhead();
                break;
            case FIRE_AUTO:
                debug("Executing action: " + action + " (fireAuto)");
                fireAuto();
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
            case AIM_GUN:
                debug("Executing action: " + action + " (aim)");
                aim();
                break;
            default:
                System.out.println("Unknown action: " + action);
                System.exit(0);
                break;
        }
        robot.setTurnRadarRight(90);
    }
    

    public void doNothing() {}

    public void facingEnemy() {
        if (robot.currentState.enemyDistance != 0) {
            double turn = robot.currentState.enemyBearing;
            System.out.println("Turn: " + turn);
            robot.setTurnRight(normalizeBearing(turn));
        }
    }
    
    public void ninetyDegrees() {
        if (robot.currentState.enemyDistance != 0) {
            double degdiff = robot.currentState.enemyBearing - 90;
            System.out.println("Turn2: " + robot.currentState.enemyBearing + " " + normalizeBearing(degdiff));
            robot.setTurnRight(normalizeBearing(degdiff));
        }
    }   

    public void runClockwise() {
        robot.setTurnRight(90);  
        robot.setAhead(50);
    }   

    public void runCounterClockwise() {
        robot.setTurnLeft( 90);   
        robot.setAhead(-50);
    }

    public void runAwayBack() {
        robot.setAhead(-50);
    }

    public void runAhead() {
        robot.setAhead(50);
    }

    public void fireAuto() {
        if (robot.currentState.enemyDistance < 100) {
            robot.setFire(3);
        } else if (robot.currentState.enemyDistance < 200) {
            robot.setFire(2);
        } else {
            robot.setFire(1);
        }
    }

    public void fire1() {
        robot.setFire(1);
    }

    public void fire2() {
        robot.setFire(2);
    }

    public void fire3() {
        robot.setFire(3);
    }

    public void aim() {
        // if (robot.currentState.enemyDistance != 0) {
        double gunTurn = robot.getHeading() - robot.getGunHeading() + robot.currentState.enemyBearing;
        robot.setTurnGunRight(normalizeBearing(gunTurn));
        // }
    }
    
    // public void turnRadarRight(double angle) {
    //     robot.setTurnRadarRight(normalizeBearing(angle));
    // }

    // public void aimRadar() {
    //     double radarTurn = robot.getHeading() - robot.getRadarHeading() + robot.currentState.enemyBearing;
    //     robot.setTurnRadarRight(normalizeBearing(radarTurn));
    // }

    double normalizeBearing(double angle) {
        while (angle >  180) angle -= 360;
        while (angle < -180) angle += 360;
        return angle;
    }

} 
