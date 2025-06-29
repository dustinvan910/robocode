package RLrobots;

import javax.websocket.*;
import java.net.URI;

@ClientEndpoint
public class WebSocketClient {

    private Session session;
    private MessageHandler messageHandler;

    // Interface for handling messages
    public interface MessageHandler {
        void handleMessage(String message);
    }

    public WebSocketClient() {
        try {
            URI endpointURI = new URI("ws://localhost:5000");
            WebSocketContainer container = ContainerProvider.getWebSocketContainer();
            container.connectToServer(this, endpointURI);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Set the message handler
    public void setMessageHandler(MessageHandler handler) {
        this.messageHandler = handler;
    }

    @OnOpen
    public void onOpen(Session session) {
        this.session = session;
        System.out.println("Connected to WebSocket");
    }

    @OnMessage
    public void onMessage(String message) {        
        // Forward the message to the handler if one is set
        if (messageHandler != null) {
            messageHandler.handleMessage(message);
        }
    }

    @OnClose
    public void onClose(Session session, CloseReason reason) {
        System.out.println("Connection closed xxx: " + reason);
    }

    public void sendMessage(String message) {
        if (session != null && session.isOpen()) {
            session.getAsyncRemote().sendText(message);
        }
    }

    public void closeConnection() throws Exception {
        System.out.println("Closing WebSocket connection");
        if (session != null && session.isOpen()) {
            session.close();
        }
    }
}