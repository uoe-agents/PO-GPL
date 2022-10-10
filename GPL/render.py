import numpy as np
import pygame
import imageio
import os
class Visualizer(object):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREY = (50,50,50)
    ORANGE = (255, 125, 0)
    WIDTH = 20
    HEIGHT = 20

    MARGIN = 0

    # Create a 2 dimensional array. A two dimensional
    # array is simply a list of lists.

    def __init__(self, grid_height=20, grid_width=20, main_sight= 2 ):
        

        self.main_sight_radius = main_sight
        self.pygame = pygame
        self.grid_height, self.grid_width = grid_height, grid_width
        # Grid to save the locations of agents, preys and obstacles
        self.grid = [[1 for b in range(self.grid_width)] for a in range(self.grid_height)]

        self.WINDOW_SIZE = [self.grid_height * self.HEIGHT, self.grid_width * self.WIDTH]

        self.pygame.init()
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        self.pygame.display.set_caption("Wolfpack")
        self.clock = pygame.time.Clock()

    def render_single(self, state_estimated, u_estimated, i_estimated, i_true, 
        current_obs, num_dones, n_updates, saving_dir, exp_name, env_name, true_existence = True , particle_num_str='average', verbose=False):
        done = False
        for event in self.pygame.event.get():  # User did something
            if event.type == self.pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        self.screen.fill(self.BLACK)
        
        if verbose:
            print('state_estimated', state_estimated.size(), u_estimated.size(), i_estimated.size())
            print(state_estimated)
            print()
        # get estimated stated, but only of estimated existing teamates
        #print(i_estimated)
        if true_existence:
            i_existence = i_true
        else:
            i_existence = i_estimated
        estate_estimation_existance_ = (i_existence*state_estimated).detach().numpy()
        estate_estimation_existance_ = np.clip(np.round(estate_estimation_existance_,0),-9.,9.).astype(np.int16)
        #print(estate_estimation_existance_)
        # exit()

        # estate_estimation_existance_ = (i_estimated*np.round(state_estimated.detach().numpy(),0)).astype(np.int16)
        # get estimated enemies
        u_estimation = np.round(u_estimated.detach().numpy(),0).astype(np.int16)
        if verbose:
            print('u estimation')
            print(u_estimation)
        # make grid black
        self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]
        
        # populate grid with agents
        n_players = 0
        for j in range(4,-1,-1): # 5 teammates go backwards so they learner is always printed
            if j == 0: # if it is the learner we plot it 
                if env_name == 'wolfpack':
                    # paint grey the sight radious
                    for i in range(self.grid_width):
                        for k in range(self.grid_height):
                            if self.is_close_manhattan_dist([estate_estimation_existance_[0,j,0],estate_estimation_existance_[0,j,1]],[i,k]):
                                if self.grid[i][k] == 0: # if there is nothing in that slot
                                    self.grid[i][k] = 8 # make it grey
                if verbose:
                    print(estate_estimation_existance_[0,j,0], estate_estimation_existance_[0,j,1])
                
                if i_existence[:,j] == 1.:
                    # make it blue if we predicted it's existence 
                    self.grid[estate_estimation_existance_[0,j,0]][estate_estimation_existance_[0,j,1]] = 4 
                    # self.grid[estate_estimation_existance_[0,j,1]][estate_estimation_existance_[0,j,0]] = 4 
                else: 
                    # otherwise make it grey
                    self.grid[estate_estimation_existance_[0,j,0]][estate_estimation_existance_[0,j,1]] = 8 
            elif i_existence[:,j] == 1.: # or better yet if we are sure it exists  
                self.grid[estate_estimation_existance_[0,j,0]][estate_estimation_existance_[0,j,1]] = 6 # make it green 
                # self.grid[estate_estimation_existance_[0,j,1]][estate_estimation_existance_[0,j,0]] = 6 # make it green 
        

        # exit()
        # populate grid with enemies
        if env_name == 'wolfpack':
            #  get the observed enemies
            observerd_enemies = np.clip(current_obs[0,4:8],-9.,9.).astype(np.int16)
            if observerd_enemies[0]>0. and observerd_enemies[1]>0.:
                self.grid[observerd_enemies[0]][observerd_enemies[1]] = 3
                #self.grid[observerd_enemies[1]][observerd_enemies[0]]
            if observerd_enemies[2]>0. and observerd_enemies[3]>0.:
                self.grid[observerd_enemies[2]][observerd_enemies[3]] = 3
                # self.grid[observerd_enemies[3]][observerd_enemies[2]] = 3

            # if estimation > 0 it means they are there 
            # clip the estimation of prey
            u_estimation = np.clip(u_estimation,-9.,9.).astype(np.int16)
            if verbose:
                print('u')
                print(u_estimation[0,0],u_estimation[0,1], u_estimation[0,2], u_estimation[0,3])
            # render them 
            if u_estimation[0,0]>0. and u_estimation[0,1]>0.:
                self.grid[u_estimation[0,0]][u_estimation[0,1]] = 5
                # self.grid[u_estimation[0,1]][u_estimation[0,0]] = 5
            if u_estimation[0,2]>0. and u_estimation[0,3]>0.:
                self.grid[u_estimation[0,2]][u_estimation[0,3]] = 5
                # self.grid[u_estimation[0,3]][u_estimation[0,2]] = 5
     	    
     

        # populated grid with seen enemies: 

        
        for row in range(self.grid_height):
            for column in range(self.grid_width):
                color = self.BLACK
                if self.grid[row][column] == 1:
                    color = self.BLACK
                elif self.grid[row][column] == 2:
                    color = self.WHITE
                #elif self.grid[row][column] == 3:
                #    color = self.RED
                elif self.grid[row][column] == 4:
                    color = self.BLUE
                elif self.grid[row][column] == 5: # estimated enemies
                    color = self.ORANGE
                elif self.grid[row][column] == 6: # estimated Teammates
                    color = self.GREEN
                elif self.grid[row][column] == 8:
                    color = self.GREY
                self.pygame.draw.rect(self.screen,
                                      color,
                                      [(self.MARGIN + self.WIDTH) * column + self.MARGIN,
                                       (self.MARGIN + self.HEIGHT) * row + self.MARGIN,
                                       self.WIDTH,
                                       self.HEIGHT])
        
        self.clock.tick(60)
        img = self.pygame.surfarray.array3d(self.screen)
        self.pygame.display.flip()

        if done:
            self.pygame.quit()
        # print("self.screen", self.pygame.surfarray.array3d(self.screen))
        self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]

        if verbose:
            print('episodes/', env_name ,exp_name ,  particle_num_str )
        directory = 'episodes/'+ env_name + '/' + exp_name + '/' + particle_num_str 


        # exit()
        if not os.path.exists(directory):
            os.makedirs(directory)

        imageio.imwrite('episodes/'+ env_name + '/' + exp_name + '/' + particle_num_str  + '/filename' + str(n_updates) +'.jpg', img)

        return 
    
    def render_every_particle(self, state_dist, u_dist, i_dist, i_true, 
        current_obs, weights, num_dones, n_updates, saving_dir, exp_name, env_name, 
        verbose=False, decode_existence=2, use_true_existence=False):
     
        s_sizes = state_dist.size()
        if verbose:
            print(s_sizes)
            print(u_dist.size())
            print(i_dist.size())

        batch_size, num_particles, num_agent, s_dim = s_sizes[0], s_sizes[1], s_sizes[2], s_sizes[-1]

        for j in range(num_particles):
            # decode existence means that to the state decoding I am adding the existence representation
            # if decode existece ==2 means I do not do it
            if decode_existence==2:
                if verbose:
                    print(state_dist[:,j])
                self.render_single(state_dist[:,j],
                                    u_dist[:,j], 
                                    i_dist[:,j],
                                    i_true, 
                                    current_obs, 
                                    num_dones, n_updates, 
                                    saving_dir, exp_name, env_name, particle_num_str= str(j), verbose=verbose, true_existence = use_true_existence)
            # if decode is 1 means I do it 
            elif decode_existence ==1:  
                # so now I have two methods to estimate the existence, and I need to shorten the state 
                if verbose:
                    print(state_dist[:,j,:,:])
                self.render_single(state_dist[:,j,:,1:3],
                                    u_dist[:,j], 
                                    i_dist[:,j],
                                    i_true, 
                                    current_obs, 
                                    num_dones, n_updates, 
                                    saving_dir, exp_name, env_name, particle_num_str= str(j), verbose=verbose, true_existence = use_true_existence)



    def is_close_manhattan_dist(self, loc1, loc2):
        return (abs(loc1[0]-loc2[0]) + abs(loc1[1]-loc2[1])) <= self.main_sight_radius


    def drawGrid(self):
        blockSize = 50  #Set the size of the grid block
        for x in range(0, self.WIDTH, blockSize):
            for y in range(0, self.HEIGHT, blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                self.pygame.draw.rect(self.screen, self.WHITE, rect, 1)

    def render_old(self):
        done = False
        for event in self.pygame.event.get():  # User did something
            if event.type == self.pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        self.screen.fill(self.BLACK)

        for row in range(self.grid_height):
            for column in range(self.grid_width):
                color = self.BLUE
                if self.grid[row][column] == 1:
                    color = self.BLACK
                elif self.grid[row][column] == 2:
                    color = self.WHITE
                elif self.grid[row][column] == 3:
                    color = self.RED
                self.pygame.draw.rect(self.screen,
                                      color,
                                      [(self.MARGIN + self.WIDTH) * column + self.MARGIN,
                                       (self.MARGIN + self.HEIGHT) * row + self.MARGIN,
                                       self.WIDTH,
                                       self.HEIGHT])

        self.clock.tick(60)
        img = self.pygame.surfarray.array3d(self.screen)
        self.pygame.display.flip()

        if done:
            self.pygame.quit()
        # print("self.screen", self.pygame.surfarray.array3d(self.screen))
        return img
    

    def render_true_state(self, state, u, existence, n_updates, directory, true_existence = True , particle_num_str='average', verbose=False):
        done = False
        for event in self.pygame.event.get():  # User did something
            if event.type == self.pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        self.screen.fill(self.BLACK)
        state = state.astype(np.int16)
        u = u[0,0,:].astype(np.int16)
        existence = existence.astype(np.int16)


        # make grid black
        self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]
        
        # populate grid with agents
        n_players = 0
        for j in range(0,5): # 5 teammates go backwards so they learner is always printed
            # if it is the learner we plot it and the sight around it 
            if j == 0: 
                # paint grey the sight radious
                for i in range(self.grid_width):
                    for k in range(self.grid_height):
                        if self.is_close_manhattan_dist([state[0,j,0],state[0,j,1]],[i,k]):
                            if self.grid[i][k] == 0: # if there is nothing in that slot
                                self.grid[i][k] = 8 # make it grey
                # paint the learner
                self.grid[state[0,j,0]][state[0,j,1]] = 4 
            elif existence[:,j] == 1.:
                #  if it is not the learner just paint it blue
                self.grid[state[0,j,0]][state[0,j,1]] = 4 
        

        # plot prey
        
        if u[0]>0. and u[1]>0.:
            self.grid[u[0]][u[1]] = 3
            #self.grid[observerd_enemies[1]][observerd_enemies[0]]
        if u[2]>0. and u[3]>0.:
            self.grid[u[2]][u[3]] = 3
            # self.grid[observerd_enemies[3]][observerd_enemies[2]] = 3

     

        # populated grid with seen enemies: 
        for row in range(self.grid_height):
            for column in range(self.grid_width):
                color = self.BLACK
                if self.grid[row][column] == 1:
                    color = self.BLACK
                elif self.grid[row][column] == 2:
                    color = self.WHITE
                elif self.grid[row][column] == 3: # Prey 
                    color = self.RED
                elif self.grid[row][column] == 4: # Teammates
                    color = self.BLUE
                elif self.grid[row][column] == 5: # estimated enemies
                    color = self.ORANGE
                elif self.grid[row][column] == 6: # estimated Teammates
                    color = self.GREEN
                elif self.grid[row][column] == 8:
                    color = self.GREY
                self.pygame.draw.rect(self.screen,
                                      color,
                                      [(self.MARGIN + self.WIDTH) * column + self.MARGIN,
                                       (self.MARGIN + self.HEIGHT) * row + self.MARGIN,
                                       self.WIDTH,
                                       self.HEIGHT])
        
        self.clock.tick(60)
        img = self.pygame.surfarray.array3d(self.screen)
        self.pygame.display.flip()

        if done:
            self.pygame.quit()
        # print("self.screen", self.pygame.surfarray.array3d(self.screen))
        self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]

        
        # exit()
        if not os.path.exists(directory):
            os.makedirs(directory)

        imageio.imwrite(directory + '/n' + str(n_updates) +'.jpg', img)

        return 