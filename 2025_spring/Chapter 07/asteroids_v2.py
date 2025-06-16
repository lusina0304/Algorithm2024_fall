# Import a library of functions called 'pygame'
import pygame
import vectors
from math import pi, sqrt, cos, sin, atan2, inf
import matrices
from random import randint, uniform
from linear_solver import do_segments_intersect, intersection
import sys

# DEFINE OBJECTS OF THE GAME

class PolygonModel():
    def __init__(self,points):
        self.points = matrices.transpose([(x, y, 1) for (x,y) in points])
        self.rotation_angle = 0
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.angular_velocity = 0

    def transformed(self):
        rotation_matrix = matrices.make_rotation_translate(self.rotation_angle, self.x, self.y)
        transformed = matrices.matrix_multiply(rotation_matrix, self.points)
        return [(x, y) for x, y, z in zip(*transformed)]

    def move(self, milliseconds):
        dx, dy = self.vx * milliseconds / 1000.0, self.vy * milliseconds / 1000.0
        self.x, self.y = vectors.add((self.x,self.y), (dx,dy))
        self.rotation_angle += self.angular_velocity * milliseconds / 1000.0

    def segments(self):
        point_count = len(self.points[0])
        points = self.transformed()
        return [(points[i], points[(i+1)%point_count])
                for i in range(0,point_count)]

    def does_collide(self, other_poly):
        for other_segment in other_poly.segments():
            if self.does_intersect(other_segment):
                return True
        return False

    def does_intersect(self, other_segment):
        for segment in self.segments():
            if do_segments_intersect(other_segment,segment):
                return True
        return False
    
    def does_intersect_with_point(self, other_segment):
        # cross_points = [inf for _ in range(len(self.points[0]))]

        # i = 0
        for segment in self.segments():
            if do_segments_intersect(other_segment,segment):
                u1,u2 = segment
                v1,v2 = other_segment

                cross = intersection(u1,u2, v1,v2)
                return cross
                # cross_points[i] = cross

        # return cross_points

    def does_intersect_with_poly(self, poly):
        for segment in self.segments():
            if poly.does_intersect(segment):
                return True
        return False


class Ship(PolygonModel):
    def __init__(self):
        super().__init__([(0.5,0), (-0.25,0.25), (-0.25,-0.25)])

    def laser_segment(self):
        dist = 20. * sqrt(2)
        x,y = self.transformed()[0]
        return (x,y), (x + dist * cos(self.rotation_angle), y + dist*sin(self.rotation_angle))


class Asteroid(PolygonModel):
    def __init__(self):
        sides = randint(5,9)
        vs = [vectors.to_cartesian((uniform(0.5,1.0), 2 * pi * i / sides))
                for i in range(0,sides)]
        super().__init__(vs)

        self.vx = uniform(-1,1)
        self.vy = uniform(-1,1)
        self.angular_velocity = uniform(-pi/2,pi/2)

# INITIALIZE GAME STATE

ship = Ship()

acceleration = 3
asteroid_count = 10
asteroids = [Asteroid() for _ in range(0,asteroid_count)]

for ast in asteroids:
    ast.x = randint(-9,9)
    ast.y = randint(-9,9)

# HELPERS / SETTINGS

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

width, height = 400, 400

def to_pixels(x,y):
    return (width/2 + width * x / 20, height/2 - height * y / 20)

def draw_poly(screen, polygon_model, color=GREEN):
    pixel_points = [to_pixels(x,y) for x,y in polygon_model.transformed()]
    pygame.draw.aalines(screen, color, True, pixel_points, 10)

def draw_poly(screen, polygon_model, center, color=GREEN):
    pixel_points = [to_pixels(x-center[0],y-center[1]) for x,y in polygon_model.transformed()]
    pygame.draw.aalines(screen, color, True, pixel_points, 10)

def draw_segment(screen, v1,v2,color=RED):
    pygame.draw.aaline(screen, color, to_pixels(*v1), to_pixels(*v2), 10)

def draw_segment(screen, v1,v2, center, color=RED):
    v1 = vectors.add(v1, vectors.scale(-1, center))
    v2 = vectors.add(v2, vectors.scale(-1, center))
    pygame.draw.aaline(screen, color, to_pixels(*v1), to_pixels(*v2), 10)


screenshot_mode = False


pygame.font.init()
myFont = pygame.font.SysFont( "arial", 30, True, False)

# INITIALIZE GAME ENGINE

def main():


    pygame.init()


    screen = pygame.display.set_mode([width,height])

    pygame.display.set_caption("Asteroids!")

    done = False
    clock = pygame.time.Clock()

    # p key prints screenshot (you can ignore this variable)
    p_pressed = False

    game_over = False

    while not done:

        clock.tick()

        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                done=True # Flag that we are done so we exit this loop

        # UPDATE THE GAME STATE

        milliseconds = clock.get_time()
        keys = pygame.key.get_pressed()

        if game_over == True:
            screen.fill(WHITE)

            for asteroid in asteroids:
                draw_poly(screen, asteroid, (ship.x, ship.y), color=GREEN)

            BLACK = ( 0, 0, 0 )
            render_text = myFont.render("GAME OVER", True, BLACK)

            text_Rect = render_text.get_rect()
            text_Rect.centerx = round(width / 2)
            text_Rect.y = 50
            
            screen.blit(render_text, text_Rect)

            if keys[pygame.K_r]:
                for ast in asteroids:
                    ast.x = randint(-9,9)
                    ast.y = randint(-9,9)

                ship.vx = 0
                ship.vy = 0
                    
                game_over = False
            
            pygame.display.flip()
            
            continue

        for ast in asteroids:
            ast.move(milliseconds)

        if keys[pygame.K_LEFT]:
            ship.rotation_angle += milliseconds * (2*pi / 1000)

        if keys[pygame.K_RIGHT]:
            ship.rotation_angle -= milliseconds * (2*pi / 1000)

        if keys[pygame.K_UP]:
            ax = acceleration * cos(ship.rotation_angle)
            ay = acceleration * sin(ship.rotation_angle)
            ship.vx += ax * milliseconds/1000
            ship.vy += ay * milliseconds/1000

        elif keys[pygame.K_DOWN]:
            ax = - acceleration * cos(ship.rotation_angle)
            ay = - acceleration * sin(ship.rotation_angle)
            ship.vx += ax * milliseconds/1000
            ship.vy += ay * milliseconds/1000

        ship.move(milliseconds)

        laser = ship.laser_segment()

        # p key saves screenshot (you can ignore this)
        if keys[pygame.K_p] and screenshot_mode:
            p_pressed = True
        elif p_pressed:
            pygame.image.save(screen, 'figures/asteroid_screenshot_%d.png' % milliseconds)
            p_pressed = False

        # DRAW THE SCENE
        screen_poly = PolygonModel([(-10,10), (10,10), (10,-10), (-10,-10)])


        screen.fill(WHITE)

        if keys[pygame.K_SPACE]:
            draw_segment(screen, *laser, (ship.x, ship.y))

        draw_poly(screen,ship, (ship.x, ship.y))

        speed = (ship.vx, ship.vy)

        for asteroid in asteroids:
            if keys[pygame.K_SPACE] and asteroid.does_intersect(laser):
                # asteroids.remove(asteroid)
                
                if vectors.length(speed) < 1e-16:
                    speed = (1,0)

                unit_dir = vectors.unit(speed)
                angle = atan2(unit_dir[1], unit_dir[0]) + uniform(-pi/2, pi/2)
                v_end = vectors.to_cartesian( (20. * sqrt(2), angle) )

                cross = screen_poly.does_intersect_with_point(((0,0), v_end))
                asteroid.x = cross[0] + ship.x
                asteroid.y = cross[1] + ship.y

            else:
                draw_poly(screen, asteroid, (ship.x, ship.y), color=GREEN)

            # redraw
            v_ast = vectors.add((asteroid.x, asteroid.y), vectors.scale(-1, (ship.x, ship.y)))

            if v_ast[0] < -10 or v_ast[0] > 10 or v_ast[1] < -10 or v_ast[1] > 10:
                if vectors.length(speed) < 1e-16:
                    speed = (1,0)

                unit_dir = vectors.unit(speed)
                angle = atan2(unit_dir[1], unit_dir[0]) + uniform(-pi/2, pi/2)
                v_end = vectors.to_cartesian( (20. * sqrt(2), angle) )

                cross = screen_poly.does_intersect_with_point(((0,0), v_end))

                # print(cross)
                # if cross != inf:

                asteroid.x = cross[0] + ship.x
                asteroid.y = cross[1] + ship.y

            #check intersect with ship
            if asteroid.does_intersect_with_poly(ship):
                game_over = True





        pygame.display.flip()




    pygame.quit()

if __name__ == "__main__":
    if '--screenshot' in sys.argv:
        screenshot_mode = True
    main()
