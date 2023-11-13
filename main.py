import pygame
import sys
import random
import math

pygame.init()

# Variabele
width, height = 1200, 800       # Breedte en hoogte van scherm
fps = 60                        # Frames per second
radius = 5                      # Radius van de bolletjes
S, I, Z, R = 500, 0, 1, 0       # Aantal mensen, I is nu nutteloos
bg_colour = (0, 0, 0)        # Background colour
direction_update = 5            # Elke 5 frames wordt de direction geupdate anders ziet het er vreemd uit (zet maar op 1)
infection_distance = 20         # Zombies kunnen susceptibles op 10 pixel afstand infecteren


class Human:
    def __init__(self, x, y, vx, vy, colour):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.colour = colour
        self.direction_counter = 0

    def move(self):     # Alle mensen gaan nu volgens dezelfde bewegingsregels bewegen, later wordt dit anders
        if self.direction_counter == direction_update:
            self.vx = random.uniform(-1, 1)
            self.vy = random.uniform(-1, 1)

            magnitude = (self.vx**2 + self.vy**2)**0.5  # De nieuwe richting moet wel dezelfde snelheid hebben
            if magnitude != 0:
                self.vx = self.vx / magnitude           # Die wordt hier genormaliseerd
                self.vy = self.vy / magnitude

            self.direction_counter = 0
        else:
            self.direction_counter += 1

        self.x += self.vx
        self.y += self.vy

        if self.x - self.radius <= 0 or self.x + self.radius >= width:      # Als ze links of rechts botsen
            self.vx = -self.vx
        if self.y - self.radius <= 0 or self.y + self.radius >= height:     # Als ze onder of boven botsen
            self.vy = -self.vy

    def draw(self, screen):
        pygame.draw.circle(screen, self.colour, (int(self.x), int(self.y)), self.radius)


class Susceptible(Human):
    def __init__(self, x, y):
        super().__init__(x, y, 0, 0, (255, 255, 255))


class Infected(Human):
    def __init__(self, x, y):
        super().__init__(x, y, 0, 0, (255, 255, 0))


class Zombie(Human):
    def __init__(self, x, y):
        super().__init__(x, y, 0, 0, (0, 255, 0))

    def infect(self, susceptible):
        distance = math.sqrt((self.x - susceptible.x)**2 + (self.y - susceptible.y)**2)
        if distance < infection_distance:
            return Infected(susceptible.x, susceptible.y)
        return None


class Resistant(Human):
    def __init__(self, x, y):
        super().__init__(x, y, 0, 0, (255, 20, 147))


susceptibles = [Susceptible(random.randint(radius, width - radius), random.randint(radius, height - radius)) for _ in range(S)]
infected = list()
zombies = [Zombie(random.randint(radius, width - radius), random.randint(radius, height - radius)) for _ in range(Z)]
resistants = [Resistant(random.randint(radius, width - radius), random.randint(radius, height - radius)) for _ in range(R)]

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Zombie Apocalypse")
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for human in susceptibles + infected + zombies + resistants:    # Alle mensen een bewegingsupdate geven
        human.move()

    new_infected = []

    for zombie in zombies:
        for susceptible in susceptibles:
            infected_individual = zombie.infect(susceptible)        # Itereren over alle zombies en susceptibles voor geinfecteerde
            if infected_individual:
                new_infected.append(infected_individual)            # Geinfecteerde mensen toevoegen aan nieuw geinfecteerde
                susceptibles.remove(susceptible)                    # En verwijderen van susceptibles

    infected.extend(new_infected)                                   # Nieuwe geinfecteerde toevoegen aan geinfecteerde

    screen.fill(bg_colour)
    for human in susceptibles + infected + zombies + resistants:    # Alle mensen tekenen
        human.draw(screen)

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
sys.exit()
