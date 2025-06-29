package RLrobots;

// State structure to hold robot state information
public class RobotState {
    public double x;
    public double y;
    public double heading;
    public double energy;
    public double gunHeading;
    public double gunHeat;
    public double velocity;
    public double distanceRemaining;

    public double enemyBearing;
    public double enemyDistance;
    public double enemyHeading;
    public double enemyX;
    public double enemyY;
    
    public long reward;
    public long time;


    public RobotState(long time) {
        // System.out.println("New state");
        // Initialize with default values and specific time
        this.x = 0;
        this.y = 0;
        this.energy = 100;
        this.velocity = 0;
        this.gunHeat = 0;
        this.heading = 0;
        this.distanceRemaining = 0;

        this.enemyBearing = 0;
        this.enemyDistance = 0;
        this.enemyHeading = 0;
        this.enemyX = 0;
        this.enemyY = 0;

        this.reward = 0;
        this.time = time;
    }
    
    public void updateRobotState(double x, double y, double heading, double energy, double gunHeading, double gunHeat, double velocity, double distanceRemaining) {
        // System.out.println("Updating robot state");
        this.x = x;
        this.y = y;
        this.heading = heading;
        this.energy = energy;
        this.gunHeading = gunHeading;
        this.gunHeat = gunHeat;
        this.velocity = velocity;
        this.distanceRemaining = distanceRemaining;
    }
    
    public void updateEnemyState(double bearing, double distance, double heading, double x, double y) {
        // System.out.println("Updating enemy state");
        this.enemyBearing = bearing;
        this.enemyDistance = distance;
        this.enemyHeading = heading;
        this.enemyX = x;
        this.enemyY = y;
    }

    public void addReward(long reward, String reason) {
        // System.out.println("Adding reward: " + reward + " - " + reason);
        this.reward += reward;
    }
    
    public String toJson() {
        return String.format(
            "{\"x\":%.2f,\"y\":%.2f,\"heading\":%.2f,\"energy\":%.2f,\"gunHeading\":%.2f,\"gunHeat\":%.2f," +
            "\"velocity\":%.2f,\"distanceRemaining\":%.2f," +
            "\"enemyBearing\":%.2f,\"enemyDistance\":%.2f,\"enemyHeading\":%.2f," +
            "\"enemyX\":%.2f,\"enemyY\":%.2f,\"reward\":%d}",
            x, y, heading, energy, gunHeading, gunHeat, velocity, distanceRemaining,
            enemyBearing, enemyDistance, enemyHeading, enemyX, enemyY, reward
        );
    }
}
