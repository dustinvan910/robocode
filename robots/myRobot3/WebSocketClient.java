package myRobot3;

import javax.websocket.*;
import java.net.URI;

@ClientEndpoint
public class WebSocketClient {

    private Session session;

    public WebSocketClient(URI endpointURI) {
        try {
            WebSocketContainer container = ContainerProvider.getWebSocketContainer();
            container.connectToServer(this, endpointURI);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @OnOpen
    public void onOpen(Session session) {
        this.session = session;
        System.out.println("Connected to WebSocket");
    }

    @OnMessage
    public void onMessage(String message) {
        System.out.println("Received: " + message);
    }

    @OnClose
    public void onClose(Session session, CloseReason reason) {
        System.out.println("Connection closed: " + reason);
    }

    public void sendMessage(String message) {
        session.getAsyncRemote().sendText(message);
    }

    public void close() throws Exception {
        System.out.println("Close connection");
        session.close();
    }
}